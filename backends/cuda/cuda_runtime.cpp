#include "cuda_runtime.h"
#include "../../runtime/kv_slot_guard.h"
#include "../../runtime/cur_pos_guard.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <regex>
#include <unordered_map>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace ember {

// =============================================================================
// 辅助函数（在 ember 命名空间）
// =============================================================================

static std::string format_bytes(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + "." +
               std::to_string((bytes / (1024 * 1024 * 10)) % 100) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + "." +
               std::to_string((bytes / (1024 * 10)) % 100) + " MB";
    } else {
        return std::to_string(bytes / 1024) + " KB";
    }
}

static cudaDataType_t to_cuda_dtype(DType dtype) {
    switch (dtype) {
        case DType::F16:  return CUDA_R_16F;
        case DType::BF16: return CUDA_R_16BF;
        case DType::F32:  return CUDA_R_32F;
        default:          return CUDA_R_16F;
    }
}

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t round_bias = 0x7FFF + lsb;
    bits += round_bias;
    return static_cast<uint16_t>(bits >> 16);
}

static void bf16_to_fp16(const uint16_t* src, half* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = __float2half_rn(bf16_to_f32(src[i]));
    }
}

static void f32_to_fp16(const float* src, half* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = __float2half_rn(src[i]);
    }
}

static void f16_to_bf16(const half* src, uint16_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = f32_to_bf16(__half2float(src[i]));
    }
}

static void f32_to_bf16(const float* src, uint16_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = f32_to_bf16(src[i]);
    }
}

static Error dump_last_row(const std::string& dump_dir, const std::string& name,
                           int device_id, const void* data, int seq_len,
                           int row_size, DType dtype) {
    if (seq_len <= 0 || row_size <= 0) {
        return Error::success();
    }
    
    size_t elem_size = dtype_size(dtype);
    if (elem_size == 0) {
        return Error(ErrorCode::INVALID_FORMAT, "Unsupported dtype for dump");
    }
    
    size_t offset = static_cast<size_t>(seq_len - 1) * row_size * elem_size;
    const char* base = static_cast<const char*>(data);
    const void* src = base + offset;
    
    std::vector<float> out(static_cast<size_t>(row_size));
    
    if (dtype == DType::F16) {
        std::vector<half> tmp(static_cast<size_t>(row_size));
        Error err = cuda::cuda_memcpy_d2h(tmp.data(), src, tmp.size() * sizeof(half), device_id);
        if (err) return err;
        for (size_t i = 0; i < tmp.size(); ++i) {
            out[i] = __half2float(tmp[i]);
        }
    } else if (dtype == DType::BF16) {
        std::vector<uint16_t> tmp(static_cast<size_t>(row_size));
        Error err = cuda::cuda_memcpy_d2h(tmp.data(), src, tmp.size() * sizeof(uint16_t), device_id);
        if (err) return err;
        for (size_t i = 0; i < tmp.size(); ++i) {
            out[i] = bf16_to_f32(tmp[i]);
        }
    } else if (dtype == DType::F32) {
        Error err = cuda::cuda_memcpy_d2h(out.data(), src, out.size() * sizeof(float), device_id);
        if (err) return err;
    } else {
        return Error(ErrorCode::INVALID_FORMAT, "Unsupported dtype for dump");
    }
    
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(dump_dir, ec);
    if (ec) {
        return Error(ErrorCode::FILE_WRITE_ERROR, "Failed to create dump dir: " + dump_dir);
    }
    
    fs::path path = fs::path(dump_dir) / (name + ".bin");
    std::ofstream out_file(path, std::ios::binary);
    if (!out_file.is_open()) {
        return Error(ErrorCode::FILE_WRITE_ERROR, "Failed to open dump file: " + path.string());
    }
    out_file.write(reinterpret_cast<const char*>(out.data()),
                   static_cast<std::streamsize>(out.size() * sizeof(float)));
    
    return Error::success();
}

static Error load_tensor_to_device(ModelWeightLoader& loader,
                                   const std::string& name,
                                   int device_id,
                                   DType target_dtype,
                                   void** ptr) {
    auto meta = loader.get_meta(name);
    if (!meta) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Missing: " + name);
    }
    
    size_t elem_size = dtype_size(meta->dtype);
    if (elem_size == 0 || meta->data_size % elem_size != 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid tensor dtype for " + name + ": " + std::string(dtype_name(meta->dtype)));
    }
    
    size_t count = meta->data_size / elem_size;
    std::vector<char> cpu_buf(meta->data_size);
    Error err = loader.read_tensor(name, cpu_buf.data(), meta->data_size);
    if (err) return err;
    
    const void* src = cpu_buf.data();
    size_t dst_size = meta->data_size;
    std::vector<half> fp16_buf;
    std::vector<uint16_t> bf16_buf;
    
    if (meta->dtype == target_dtype) {
        // 原样拷贝
    } else if (target_dtype == DType::F16) {
        fp16_buf.resize(count);
        if (meta->dtype == DType::BF16) {
            bf16_to_fp16(reinterpret_cast<const uint16_t*>(cpu_buf.data()), fp16_buf.data(), count);
        } else if (meta->dtype == DType::F32) {
            f32_to_fp16(reinterpret_cast<const float*>(cpu_buf.data()), fp16_buf.data(), count);
        } else {
            return Error(ErrorCode::INVALID_FORMAT,
                         "Unsupported tensor dtype for " + name + ": " + std::string(dtype_name(meta->dtype)));
        }
        src = fp16_buf.data();
        dst_size = count * sizeof(half);
    } else if (target_dtype == DType::BF16) {
        bf16_buf.resize(count);
        if (meta->dtype == DType::F16) {
            f16_to_bf16(reinterpret_cast<const half*>(cpu_buf.data()), bf16_buf.data(), count);
        } else if (meta->dtype == DType::F32) {
            f32_to_bf16(reinterpret_cast<const float*>(cpu_buf.data()), bf16_buf.data(), count);
        } else {
            return Error(ErrorCode::INVALID_FORMAT,
                         "Unsupported tensor dtype for " + name + ": " + std::string(dtype_name(meta->dtype)));
        }
        src = bf16_buf.data();
        dst_size = count * sizeof(uint16_t);
    } else {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Unsupported target dtype for " + name + ": " + std::string(dtype_name(target_dtype)));
    }
    
    err = cuda::cuda_malloc(ptr, dst_size, device_id);
    if (err) return err;
    
    return cuda::cuda_memcpy_h2d(*ptr, src, dst_size, device_id);
}

static Error copy_bytes_peer_or_staged(void* dst, int dst_device,
                                       const void* src, int src_device,
                                       size_t bytes) {
    // Try peer copy first (fast path). If it fails, fall back to chunked staging via pinned host.
    cudaError_t err = cudaMemcpyPeer(dst, dst_device, src, src_device, bytes);
    if (err == cudaSuccess) return Error::success();
    cudaGetLastError();

    constexpr size_t kChunk = 64ull * 1024ull * 1024ull;
    void* host = nullptr;
    size_t buf = std::min(kChunk, bytes);
    cudaError_t herr = cudaMallocHost(&host, buf);
    if (herr != cudaSuccess) {
        return Error::cuda_error(std::string("cudaMallocHost failed: ") + cudaGetErrorString(herr));
    }

    const char* src_bytes = static_cast<const char*>(src);
    char* dst_bytes = static_cast<char*>(dst);
    for (size_t off = 0; off < bytes; off += buf) {
        size_t n = std::min(buf, bytes - off);
        Error e = cuda::cuda_memcpy_d2h(host, src_bytes + off, n, src_device);
        if (e) {
            cudaFreeHost(host);
            return e;
        }
        e = cuda::cuda_memcpy_h2d(dst_bytes + off, host, n, dst_device);
        if (e) {
            cudaFreeHost(host);
            return e;
        }
    }

    cudaFreeHost(host);
    return Error::success();
}

struct LoraTargetKey {
    int layer_idx = -1;
    std::string proj;
    bool is_a = false;
};

static bool parse_lora_target_key(const std::string& tensor_name, LoraTargetKey& out) {
    // Supports:
    //   ...layers.<i>.self_attn.<proj>.lora_A.weight
    //   ...layers.<i>.self_attn.<proj>.lora_B.weight
    //   ...layers.<i>.self_attn.<proj>.lora_A.default.weight
    //   ...layers.<i>.self_attn.<proj>.lora_B.default.weight
    static const std::regex kPat(
        R"(layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.default)?\.weight$)"
    );
    std::smatch m;
    if (!std::regex_search(tensor_name, m, kPat)) {
        return false;
    }
    out.layer_idx = std::stoi(m[1].str());
    out.proj = m[2].str();
    out.is_a = (m[3].str() == "A");
    return true;
}

static float read_lora_alpha_over_r(const std::string& adapter_dir) {
    namespace fs = std::filesystem;
    const fs::path cfg = fs::path(adapter_dir) / "adapter_config.json";
    if (!fs::exists(cfg)) return 1.0f;
    std::ifstream in(cfg);
    if (!in.is_open()) return 1.0f;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (s.empty()) return 1.0f;

    std::smatch m_alpha;
    std::smatch m_r;
    static const std::regex kAlpha(R"("lora_alpha"\s*:\s*([0-9]+(?:\.[0-9]+)?))");
    static const std::regex kR(R"("r"\s*:\s*([0-9]+(?:\.[0-9]+)?))");
    if (!std::regex_search(s, m_alpha, kAlpha)) return 1.0f;
    if (!std::regex_search(s, m_r, kR)) return 1.0f;

    const float alpha = std::stof(m_alpha[1].str());
    const float r = std::stof(m_r[1].str());
    if (alpha <= 0.0f || r <= 0.0f) return 1.0f;
    return alpha / r;
}

std::string MemoryEstimate::to_string() const {
    std::string s;
    s += "Memory Estimate:\n";
    s += "  Weights:     " + format_bytes(weights_bytes) + "\n";
    s += "  KV Cache:    " + format_bytes(kv_cache_bytes) + "\n";
    s += "  Activations: " + format_bytes(activation_bytes) + "\n";
    s += "  Workspace:   " + format_bytes(workspace_bytes) + "\n";
    s += "  Total:       " + format_bytes(total_bytes) + "\n";
    return s;
}

void DeviceMap::print() const {
    std::cout << "Device Map:\n";
    std::cout << "  Embedding: GPU " << embedding_device << "\n";
    std::cout << "  LM Head:   GPU " << lm_head_device << "\n";
    std::cout << "  Layers:\n";
    
    int prev_device = -1;
    int start_layer = 0;
    for (size_t i = 0; i <= layer_to_device.size(); ++i) {
        int device = (i < layer_to_device.size()) ? layer_to_device[i] : -1;
        if (device != prev_device) {
            if (prev_device >= 0) {
                std::cout << "    Layers " << start_layer << "-" << (i-1) 
                          << " -> GPU " << prev_device << "\n";
            }
            start_layer = i;
            prev_device = device;
        }
    }
}

DeviceMap DeviceMap::auto_map(const ModelConfig& config,
                              const std::vector<size_t>& gpu_free_memory,
                              int ctx_len, int batch_size) {
    DeviceMap dm;
    int num_gpus = static_cast<int>(gpu_free_memory.size());
    int num_layers = config.num_layers;
    
    if (num_gpus == 0) {
        // 没有 GPU，返回空映射
        return dm;
    }
    
    if (num_gpus == 1) {
        // 单卡，所有层都在 GPU 0
        return single_device(num_layers, 0);
    }

    // ---------------------------------------------------------------------
    // 多卡：以“每层显存”作为单位，做连续分段切分（更适合 pipeline）。
    // 注意：这里是估算（权重 + KV），用于初步切分避免 OOM，精确值以实际分配为准。
    // ---------------------------------------------------------------------
    DType dtype = dtype_from_string(config.torch_dtype);
    if (dtype == DType::UNKNOWN) dtype = DType::F16;
    const size_t elem = dtype_size(dtype);

    auto estimate_layer_weight_bytes = [&]() -> size_t {
        const size_t H = static_cast<size_t>(config.hidden_size);
        const size_t I = static_cast<size_t>(config.intermediate_size);
        const size_t Nh = static_cast<size_t>(config.num_heads);
        const size_t Nk = static_cast<size_t>(config.num_kv_heads);
        const size_t Hd = static_cast<size_t>(config.head_dim);

        size_t total = 0;
        total += H * (Nh * Hd) * elem;   // q_proj
        total += H * (Nk * Hd) * elem;   // k_proj
        total += H * (Nk * Hd) * elem;   // v_proj
        total += (Nh * Hd) * H * elem;   // o_proj
        total += (Hd * 2) * elem;        // q_norm/k_norm
        total += (H * I) * elem;         // gate_proj
        total += (H * I) * elem;         // up_proj
        total += (I * H) * elem;         // down_proj
        total += (H * 2) * elem;         // layernorm weights (approx)
        return total;
    };

    const size_t per_layer_weights = estimate_layer_weight_bytes();
    const size_t per_layer_kv = config.kv_cache_size_per_layer(ctx_len, batch_size, dtype);
    const size_t per_layer_total = per_layer_weights + per_layer_kv;

    // 额外权重：embedding / lm_head / final_norm
    const size_t embed_bytes = static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size) * elem;
    const size_t final_norm_bytes = static_cast<size_t>(config.hidden_size) * elem;
    const size_t lm_head_bytes = config.tie_word_embeddings ? embed_bytes
                                                            : (static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size) * elem);

    std::vector<size_t> avail = gpu_free_memory;
    if (!avail.empty()) {
        avail[0] = (avail[0] > embed_bytes) ? (avail[0] - embed_bytes) : 0;
    }
    if (avail.size() >= 2) {
        size_t& last = avail[avail.size() - 1];
        const size_t extra = final_norm_bytes + lm_head_bytes;
        last = (last > extra) ? (last - extra) : 0;
    }

    std::vector<size_t> capacity(num_gpus, 0);
    size_t cap_sum = 0;
    for (int i = 0; i < num_gpus; ++i) {
        capacity[i] = (per_layer_total == 0) ? 0 : (avail[i] / per_layer_total);
        cap_sum += capacity[i];
    }

    dm.layer_to_device.resize(num_layers);
    dm.num_devices = num_gpus;
    dm.embedding_device = 0;
    dm.lm_head_device = num_gpus - 1;

    // 容量估算不足：退化为均分（由后续实际分配决定是否 OOM）。
    if (cap_sum < static_cast<size_t>(num_layers) || per_layer_total == 0) {
        int current = 0;
        const int layers_per_device = (num_layers + num_gpus - 1) / num_gpus;
        for (int l = 0; l < num_layers; ++l) {
            dm.layer_to_device[l] = current;
            if ((l + 1) % layers_per_device == 0 && current < num_gpus - 1) current++;
        }
        return dm;
    }

    // 目标层数：按 avail 比例分配，并限制在 capacity 内。
    const double total_avail = std::accumulate(avail.begin(), avail.end(), 0.0);
    std::vector<int> layers_on_gpu(num_gpus, 0);
    for (int i = 0; i < num_gpus; ++i) {
        int want = static_cast<int>(std::floor((avail[i] / total_avail) * num_layers));
        want = std::max(want, 1);
        want = std::min<int>(want, static_cast<int>(capacity[i]));
        layers_on_gpu[i] = want;
    }

    auto sum_layers = [&]() -> int {
        int s = 0;
        for (int v : layers_on_gpu) s += v;
        return s;
    };

    // 调整到恰好 num_layers。
    while (sum_layers() > num_layers) {
        int idx = std::max_element(layers_on_gpu.begin(), layers_on_gpu.end()) - layers_on_gpu.begin();
        if (layers_on_gpu[idx] > 1) layers_on_gpu[idx]--;
        else break;
    }
    while (sum_layers() < num_layers) {
        int best = -1;
        size_t best_slack = 0;
        for (int i = 0; i < num_gpus; ++i) {
            if (layers_on_gpu[i] >= static_cast<int>(capacity[i])) continue;
            const size_t used = static_cast<size_t>(layers_on_gpu[i]) * per_layer_total;
            const size_t slack = (avail[i] > used) ? (avail[i] - used) : 0;
            if (best < 0 || slack > best_slack) {
                best = i;
                best_slack = slack;
            }
        }
        if (best < 0) break;
        layers_on_gpu[best]++;
    }

    // 连续分段写入 layer_to_device。
    int layer = 0;
    for (int dev = 0; dev < num_gpus; ++dev) {
        for (int k = 0; k < layers_on_gpu[dev] && layer < num_layers; ++k) {
            dm.layer_to_device[layer++] = dev;
        }
    }
    while (layer < num_layers) dm.layer_to_device[layer++] = num_gpus - 1;

    return dm;
}

// =============================================================================
// CudaRuntime 实现
// =============================================================================

namespace cuda {

CudaRuntime::CudaRuntime() {
    // 初始化
}

CudaRuntime::~CudaRuntime() {
    unload();
}

bool CudaRuntime::available() const {
    return get_device_count() > 0;
}

static void try_enable_peer_access_all_pairs(int num_devices) {
    for (int i = 0; i < num_devices; ++i) {
        for (int j = 0; j < num_devices; ++j) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);
            if (!can) continue;
            cudaSetDevice(i);
            cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
                continue;
            }
            if (err != cudaSuccess) {
                cudaGetLastError();
            }
        }
    }
}

MemoryEstimate CudaRuntime::estimate_memory(const ModelConfig& config,
                                             int ctx_len, int batch_size) {
    MemoryEstimate est;
    DType estimate_dtype = dtype_from_string(config.torch_dtype);
    if (estimate_dtype == DType::UNKNOWN) {
        estimate_dtype = DType::F16;
    }
    size_t elem_size = dtype_size(estimate_dtype);
    
    // 权重大小
    est.weights_bytes = config.estimate_weights_size(estimate_dtype);
    
    // KV Cache
    est.kv_cache_bytes = config.kv_cache_size_per_layer(ctx_len, batch_size, estimate_dtype) 
                         * config.num_layers;
    
    // 激活值（估算峰值）
    size_t hidden_size = config.hidden_size;
    size_t intermediate_size = config.intermediate_size;
    size_t num_heads = config.num_heads;
    size_t head_dim = config.head_dim;
    
    // hidden_states + norm_out + QKV outputs + attn_out + MLP buffers
    est.activation_bytes = batch_size * ctx_len * hidden_size * elem_size * 4;  // 主要 buffers
    est.activation_bytes += batch_size * ctx_len * (num_heads * head_dim * 3) * elem_size;  // QKV
    est.activation_bytes += batch_size * ctx_len * intermediate_size * elem_size * 2;  // MLP
    est.activation_bytes += batch_size * num_heads * ctx_len * ctx_len * sizeof(float);  // attn scores (FP32)
    est.activation_bytes += batch_size * num_heads * ctx_len * ctx_len * elem_size;      // attn probs
    
    // Workspace（cuBLAS 等）
    est.workspace_bytes = 256 * 1024 * 1024;  // 256 MB
    
    est.compute_total();
    return est;
}

Error CudaRuntime::load(const std::string& model_path,
                        const ModelConfig& config,
                        const DeviceMap& device_map) {
    if (loaded_) {
        unload();
    }
    
    config_ = config;
    device_map_ = device_map;
    
    std::cout << "[CudaRuntime] Loading model from: " << model_path << std::endl;
    std::cout << "[CudaRuntime] Model: " << config.model_type << std::endl;
    std::cout << "[CudaRuntime] Layers: " << config.num_layers << std::endl;
    std::cout << "[CudaRuntime] Hidden size: " << config.hidden_size << std::endl;
    
    device_map_.print();
    
    // 初始化 cuBLAS 和 streams
    int num_devices = device_map_.num_devices;
    cublas_handles_.resize(num_devices);
    streams_.resize(num_devices);
    transfer_streams_.resize(num_devices);
    profile_events_.resize(num_devices);
    
    for (int i = 0; i < num_devices; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        CUDA_CHECK(cudaStreamCreate(&transfer_streams_[i]));
        Error err = cublas_handles_[i].create(i);
        if (err) return err;
        CUDA_CHECK(cudaEventCreate(&profile_events_[i].start));
        CUDA_CHECK(cudaEventCreate(&profile_events_[i].end));
    }

    try_enable_peer_access_all_pairs(num_devices);
    
    // 加载权重
    Error err = load_weights(model_path);
    if (err) {
        unload();
        return err;
    }
    
    loaded_ = true;
    std::cout << "[CudaRuntime] Model loaded successfully" << std::endl;
    
    // 打印显存使用
    for (int i = 0; i < num_devices; ++i) {
        print_memory_usage(i);
    }
    
    return Error::success();
}

Error CudaRuntime::load_weights(const std::string& model_path) {
    ModelWeightLoader loader;
    Error err = loader.open(model_path);
    if (err) return err;
    
    // 检查是否有必要的权重
    if (!loader.has_tensor("model.embed_tokens.weight")) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Missing embedding weights");
    }
    auto embed_meta = loader.get_meta("model.embed_tokens.weight");
    if (!embed_meta) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Missing embedding weights");
    }
    
    if (embed_meta->dtype == DType::BF16 || embed_meta->dtype == DType::F16) {
        weights_.dtype = embed_meta->dtype;
    } else {
        weights_.dtype = DType::F16;
    }
    
    int device_id = device_map_.embedding_device;
    weights_.embed_device_id = device_id;
    
    // 加载 embedding
    {
        const std::string name = "model.embed_tokens.weight";
        auto meta = loader.get_meta(name);
        size_t size = meta ? meta->data_size : 0;
        err = load_tensor_to_device(loader, name, device_id, weights_.dtype, &weights_.embed_tokens);
        if (err) return err;
        
        std::cout << "[CudaRuntime] Loaded embed_tokens: " << format_bytes(size) 
                  << " -> GPU " << device_id << std::endl;
    }
    
    // 加载每层权重
    weights_.layers.resize(config_.num_layers);
    
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        int layer_device = device_map_.layer_to_device[layer_idx];
        err = load_layer_weights(layer_idx, loader, layer_device);
        if (err) return err;
        
        if ((layer_idx + 1) % 10 == 0 || layer_idx == config_.num_layers - 1) {
            std::cout << "[CudaRuntime] Loaded layers 0-" << layer_idx << std::endl;
        }
    }
    
    // 加载 final norm
    {
        const std::string name = "model.norm.weight";
        int lm_device = device_map_.lm_head_device;
        err = load_tensor_to_device(loader, name, lm_device, weights_.dtype, &weights_.final_norm);
        if (err) return err;
        weights_.final_norm_device_id = lm_device;
    }
    
    // 加载 lm_head（如果不与 embedding 共享）
    if (loader.has_tensor("lm_head.weight")) {
        const std::string name = "lm_head.weight";
        int lm_device = device_map_.lm_head_device;
        err = load_tensor_to_device(loader, name, lm_device, weights_.dtype, &weights_.lm_head);
        if (err) return err;
        weights_.lm_head_device_id = lm_device;
        weights_.lm_head_owns_allocation = true;
        std::cout << "[CudaRuntime] Loaded separate lm_head" << std::endl;
    } else {
        // 共享 embedding（若 lm_head 在不同 GPU，需要复制一份 embedding 权重到 lm_head GPU）
        int lm_device = device_map_.lm_head_device;
        weights_.lm_head_device_id = device_id;
        weights_.lm_head_owns_allocation = false;
        weights_.lm_head = weights_.embed_tokens;

        if (lm_device != device_id) {
            const size_t bytes = static_cast<size_t>(config_.vocab_size) *
                                 static_cast<size_t>(config_.hidden_size) *
                                 dtype_size(weights_.dtype);
            void* lm_copy = nullptr;
            err = cuda::cuda_malloc(&lm_copy, bytes, lm_device);
            if (err) return err;
            err = copy_bytes_peer_or_staged(lm_copy, lm_device, weights_.embed_tokens, device_id, bytes);
            if (err) return err;
            weights_.lm_head = lm_copy;
            weights_.lm_head_device_id = lm_device;
            weights_.lm_head_owns_allocation = true;
            std::cout << "[CudaRuntime] Copied tied lm_head to GPU " << lm_device << std::endl;
        } else {
            std::cout << "[CudaRuntime] Using tied embeddings for lm_head" << std::endl;
        }
    }
    
    return Error::success();
}

Error CudaRuntime::load_layer_weights(int layer_idx, ModelWeightLoader& loader, int device_id) {
    auto& layer = weights_.layers[layer_idx];
    layer.device_id = device_id;
    
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    auto load_weight = [&](const std::string& name, void** ptr) -> Error {
        return load_tensor_to_device(loader, prefix + name, device_id, weights_.dtype, ptr);
    };
    
    // Self Attention
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.q_proj.weight", &layer.q_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.k_proj.weight", &layer.k_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.v_proj.weight", &layer.v_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.o_proj.weight", &layer.o_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.q_norm.weight", &layer.q_norm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.k_norm.weight", &layer.k_norm_weight));
    
    // MLP gate/up packed contiguously: [2 * intermediate_size, hidden_size]
    // This enables decode fast path with one strided-batched GEMM for gate+up.
    const std::string gate_name = prefix + "mlp.gate_proj.weight";
    const std::string up_name = prefix + "mlp.up_proj.weight";
    const SafetensorsMeta* gate_meta = loader.get_meta(gate_name);
    const SafetensorsMeta* up_meta = loader.get_meta(up_name);
    if (!gate_meta || !up_meta) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND,
                     "Missing MLP gate/up weight at layer " + std::to_string(layer_idx));
    }
    if (gate_meta->shape != up_meta->shape) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "MLP gate/up shape mismatch at layer " + std::to_string(layer_idx));
    }

    const size_t gate_elem_size = dtype_size(gate_meta->dtype);
    const size_t up_elem_size = dtype_size(up_meta->dtype);
    if (gate_elem_size == 0 || up_elem_size == 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid MLP gate/up dtype at layer " + std::to_string(layer_idx));
    }
    if (gate_meta->data_size % gate_elem_size != 0 || up_meta->data_size % up_elem_size != 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid MLP gate/up tensor size at layer " + std::to_string(layer_idx));
    }

    const size_t gate_count = gate_meta->data_size / gate_elem_size;
    const size_t up_count = up_meta->data_size / up_elem_size;
    if (gate_count != up_count) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "MLP gate/up element count mismatch at layer " + std::to_string(layer_idx));
    }

    const size_t target_elem_size = dtype_size(weights_.dtype);
    const size_t gate_bytes = gate_count * target_elem_size;
    const size_t up_bytes = up_count * target_elem_size;
    EMBER_RETURN_IF_ERROR(cuda::cuda_malloc(&layer.gate_up_proj_weight, gate_bytes + up_bytes, device_id));

    void* gate_tmp = nullptr;
    Error err = load_tensor_to_device(loader, gate_name, device_id, weights_.dtype, &gate_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }
    err = cuda::cuda_memcpy_d2d(layer.gate_up_proj_weight, gate_tmp, gate_bytes, device_id);
    cuda_free(gate_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    void* up_tmp = nullptr;
    err = load_tensor_to_device(loader, up_name, device_id, weights_.dtype, &up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }
    void* up_dst = static_cast<void*>(static_cast<char*>(layer.gate_up_proj_weight) + gate_bytes);
    err = cuda::cuda_memcpy_d2d(up_dst, up_tmp, up_bytes, device_id);
    cuda_free(up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    layer.gate_proj_weight = layer.gate_up_proj_weight;
    layer.up_proj_weight = up_dst;
    layer.gate_up_proj_packed = true;
    EMBER_RETURN_IF_ERROR(load_weight("mlp.down_proj.weight", &layer.down_proj_weight));
    
    // LayerNorms
    EMBER_RETURN_IF_ERROR(load_weight("input_layernorm.weight", &layer.input_layernorm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("post_attention_layernorm.weight", &layer.post_attention_layernorm_weight));
    
    layer.allocated = true;
    return Error::success();
}

Error CudaRuntime::apply_lora_adapter(const std::string& adapter_dir,
                                      float scale,
                                      bool replace_existing,
                                      LoraApplyStats* stats) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (adapter_dir.empty()) {
        return Error::invalid_argument("adapter_dir is empty");
    }
    if (weights_.dtype != DType::F16 && weights_.dtype != DType::BF16) {
        return Error(ErrorCode::INVALID_FORMAT, "LoRA apply supports F16/BF16 weights only");
    }

    auto apply_one = [&](const std::string& one_adapter_dir,
                         float one_scale,
                         bool print_log,
                         LoraApplyStats* out_stats) -> Error {
        LoraApplyStats local_stats{};
        auto t0 = std::chrono::high_resolution_clock::now();

        ModelWeightLoader loader;
        Error err = loader.open(one_adapter_dir);
        if (err) return err;

        struct ABPair {
            std::string a_name;
            std::string b_name;
        };
        std::unordered_map<std::string, ABPair> pairs;
        const std::vector<std::string> names = loader.tensor_names();
        for (const std::string& name : names) {
            LoraTargetKey key;
            if (!parse_lora_target_key(name, key)) continue;
            const std::string pair_key = std::to_string(key.layer_idx) + ":" + key.proj;
            auto& p = pairs[pair_key];
            if (key.is_a) {
                p.a_name = name;
            } else {
                p.b_name = name;
            }
        }
        if (pairs.empty()) {
            return Error(ErrorCode::WEIGHT_NOT_FOUND,
                         "No supported LoRA tensors found under " + one_adapter_dir);
        }

        const float alpha_over_r = read_lora_alpha_over_r(one_adapter_dir);
        const float effective_scale = one_scale * alpha_over_r;
        local_stats.scale_used = effective_scale;

        const cudaDataType_t cuda_dtype = to_cuda_dtype(weights_.dtype);

        auto pick_target = [&](int layer_idx, const std::string& proj, void** weight_ptr,
                               int* out_dim, int* in_dim, int* device_id) -> Error {
            if (layer_idx < 0 || layer_idx >= config_.num_layers) {
                return Error(ErrorCode::INVALID_ARGUMENT,
                             "LoRA layer index out of range: " + std::to_string(layer_idx));
            }
            auto& layer = weights_.layers[static_cast<size_t>(layer_idx)];
            *device_id = layer.device_id;
            if (proj == "q_proj") {
                *weight_ptr = layer.q_proj_weight;
                *out_dim = config_.num_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "k_proj") {
                *weight_ptr = layer.k_proj_weight;
                *out_dim = config_.num_kv_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "v_proj") {
                *weight_ptr = layer.v_proj_weight;
                *out_dim = config_.num_kv_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "o_proj") {
                *weight_ptr = layer.o_proj_weight;
                *out_dim = config_.hidden_size;
                *in_dim = config_.num_heads * config_.head_dim;
            } else {
                return Error(ErrorCode::INVALID_ARGUMENT, "Unsupported LoRA target: " + proj);
            }
            if (*weight_ptr == nullptr) {
                return Error(ErrorCode::WEIGHT_NOT_FOUND,
                             "Target weight not allocated for layer " + std::to_string(layer_idx) +
                             " proj " + proj);
            }
            return Error::success();
        };

        for (const auto& kv : pairs) {
            const ABPair& p = kv.second;
            if (p.a_name.empty() || p.b_name.empty()) {
                local_stats.skipped_matrices++;
                continue;
            }

            LoraTargetKey key{};
            if (!parse_lora_target_key(p.a_name, key)) {
                local_stats.skipped_matrices++;
                continue;
            }

            const SafetensorsMeta* a_meta = loader.get_meta(p.a_name);
            const SafetensorsMeta* b_meta = loader.get_meta(p.b_name);
            if (!a_meta || !b_meta || a_meta->shape.size() != 2 || b_meta->shape.size() != 2) {
                return Error(ErrorCode::SHAPE_MISMATCH,
                             "Invalid LoRA tensor shape for pair: " + p.a_name + " / " + p.b_name);
            }

            void* weight_ptr = nullptr;
            int out_dim = 0;
            int in_dim = 0;
            int device_id = 0;
            EMBER_RETURN_IF_ERROR(
                pick_target(key.layer_idx, key.proj, &weight_ptr, &out_dim, &in_dim, &device_id));

            const int r = static_cast<int>(a_meta->shape[0]);
            const int a_in = static_cast<int>(a_meta->shape[1]);
            const int b_out = static_cast<int>(b_meta->shape[0]);
            const int b_r = static_cast<int>(b_meta->shape[1]);
            if (r <= 0 || a_in <= 0 || b_out <= 0 || b_r <= 0) {
                return Error(ErrorCode::SHAPE_MISMATCH,
                             "Non-positive LoRA dimensions for pair: " + p.a_name + " / " + p.b_name);
            }
            if (a_in != in_dim || b_out != out_dim || b_r != r) {
                return Error(ErrorCode::SHAPE_MISMATCH,
                             "LoRA shape mismatch at layer " + std::to_string(key.layer_idx) +
                             " proj " + key.proj +
                             " (A=[" + std::to_string(r) + "," + std::to_string(a_in) + "]"
                             ", B=[" + std::to_string(b_out) + "," + std::to_string(b_r) + "]"
                             ", expected out=" + std::to_string(out_dim) +
                             ", in=" + std::to_string(in_dim) + ")");
            }

            void* d_a = nullptr;
            void* d_b = nullptr;
            auto cleanup = [&]() {
                cuda_free(d_a);
                cuda_free(d_b);
                d_a = nullptr;
                d_b = nullptr;
            };

            err = load_tensor_to_device(loader, p.a_name, device_id, weights_.dtype, &d_a);
            if (err) {
                cleanup();
                return err;
            }
            err = load_tensor_to_device(loader, p.b_name, device_id, weights_.dtype, &d_b);
            if (err) {
                cleanup();
                return err;
            }

            cudaError_t cu_err = cudaSetDevice(device_id);
            if (cu_err != cudaSuccess) {
                cleanup();
                return Error::cuda_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cu_err));
            }

            cublasHandle_t handle = cublas_handles_[static_cast<size_t>(device_id)].get();
            cudaStream_t stream = streams_[static_cast<size_t>(device_id)];

            cublasStatus_t cb = cublasSetStream(handle, stream);
            if (cb != CUBLAS_STATUS_SUCCESS) {
                cleanup();
                return Error::cuda_error("cublasSetStream failed: " + std::to_string(static_cast<int>(cb)));
            }

            // Row-major update:
            //   W_row[out, in] += scale * (B_row[out, r] @ A_row[r, in])
            // Compute through column-major view:
            //   W_col[in, out] += scale * (A_col[in, r] @ B_col[r, out])
            const float alpha = effective_scale;
            const float beta = 1.0f;
            cb = cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                in_dim, out_dim, r,
                &alpha,
                d_a, cuda_dtype, in_dim,
                d_b, cuda_dtype, r,
                &beta,
                weight_ptr, cuda_dtype, in_dim,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
            if (cb != CUBLAS_STATUS_SUCCESS) {
                cleanup();
                return Error::cuda_error(
                    "cublasGemmEx (LoRA merge) failed: " + std::to_string(static_cast<int>(cb)));
            }

            cu_err = cudaStreamSynchronize(stream);
            if (cu_err != cudaSuccess) {
                cleanup();
                return Error::cuda_error(
                    std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cu_err));
            }

            cleanup();
            local_stats.updated_matrices++;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        local_stats.wall_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (local_stats.updated_matrices <= 0) {
            return Error(ErrorCode::WEIGHT_NOT_FOUND,
                         "No complete LoRA A/B pairs were applied under " + one_adapter_dir);
        }
        if (out_stats) {
            *out_stats = local_stats;
        }
        if (print_log) {
            std::cout << "[CudaRuntime] LoRA applied: updated=" << local_stats.updated_matrices
                      << ", skipped=" << local_stats.skipped_matrices
                      << ", scale=" << local_stats.scale_used
                      << ", wall_ms=" << local_stats.wall_ms << std::endl;
        }
        return Error::success();
    };

    LoraApplyStats aggregate_stats{};
    if (replace_existing && has_active_lora_adapter_) {
        LoraApplyStats rollback_stats{};
        Error rollback_err = apply_one(active_lora_adapter_dir_, -active_lora_scale_, false, &rollback_stats);
        if (rollback_err) return rollback_err;
        aggregate_stats.updated_matrices += rollback_stats.updated_matrices;
        aggregate_stats.skipped_matrices += rollback_stats.skipped_matrices;
        aggregate_stats.wall_ms += rollback_stats.wall_ms;
        has_active_lora_adapter_ = false;
        active_lora_adapter_dir_.clear();
        active_lora_scale_ = 0.0f;
    }

    LoraApplyStats local_stats{};
    Error err = apply_one(adapter_dir, scale, true, &local_stats);
    if (err) return err;
    aggregate_stats.updated_matrices += local_stats.updated_matrices;
    aggregate_stats.skipped_matrices += local_stats.skipped_matrices;
    aggregate_stats.scale_used = local_stats.scale_used;
    aggregate_stats.wall_ms += local_stats.wall_ms;

    has_active_lora_adapter_ = true;
    active_lora_adapter_dir_ = adapter_dir;
    active_lora_scale_ = scale;

    if (stats) {
        *stats = aggregate_stats;
    }
    return Error::success();
}

Error CudaRuntime::debug_copy_attention_weight(int layer_idx,
                                               const std::string& proj,
                                               std::vector<float>& out,
                                               int* out_dim,
                                               int* in_dim) const {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (layer_idx < 0 || layer_idx >= config_.num_layers) {
        return Error::invalid_argument("layer_idx out of range");
    }

    const auto& layer = weights_.layers[static_cast<size_t>(layer_idx)];
    const void* w_ptr = nullptr;
    int o_dim = 0;
    int i_dim = 0;
    if (proj == "q_proj") {
        w_ptr = layer.q_proj_weight;
        o_dim = config_.num_heads * config_.head_dim;
        i_dim = config_.hidden_size;
    } else if (proj == "k_proj") {
        w_ptr = layer.k_proj_weight;
        o_dim = config_.num_kv_heads * config_.head_dim;
        i_dim = config_.hidden_size;
    } else if (proj == "v_proj") {
        w_ptr = layer.v_proj_weight;
        o_dim = config_.num_kv_heads * config_.head_dim;
        i_dim = config_.hidden_size;
    } else if (proj == "o_proj") {
        w_ptr = layer.o_proj_weight;
        o_dim = config_.hidden_size;
        i_dim = config_.num_heads * config_.head_dim;
    } else {
        return Error::invalid_argument("unsupported proj: " + proj);
    }
    if (w_ptr == nullptr) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "target weight ptr is null");
    }

    const size_t n = static_cast<size_t>(o_dim) * static_cast<size_t>(i_dim);
    const size_t elem = dtype_size(weights_.dtype);
    if (elem == 0) {
        return Error(ErrorCode::INVALID_FORMAT, "invalid weight dtype");
    }

    std::vector<uint8_t> host_raw(n * elem);
    cudaError_t cu = cudaSetDevice(layer.device_id);
    if (cu != cudaSuccess) {
        return Error::cuda_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cu));
    }
    cu = cudaMemcpy(host_raw.data(), w_ptr, host_raw.size(), cudaMemcpyDeviceToHost);
    if (cu != cudaSuccess) {
        return Error::cuda_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(cu));
    }

    out.resize(n);
    if (weights_.dtype == DType::F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(host_raw.data());
        for (size_t idx = 0; idx < n; ++idx) {
            __half_raw raw{};
            raw.x = p[idx];
            out[idx] = __half2float(static_cast<__half>(raw));
        }
    } else if (weights_.dtype == DType::BF16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(host_raw.data());
        for (size_t idx = 0; idx < n; ++idx) {
            out[idx] = bf16_to_f32(p[idx]);
        }
    } else if (weights_.dtype == DType::F32) {
        const float* p = reinterpret_cast<const float*>(host_raw.data());
        std::copy(p, p + n, out.begin());
    } else {
        return Error(ErrorCode::INVALID_FORMAT, "unsupported weight dtype for debug copy");
    }

    if (out_dim) *out_dim = o_dim;
    if (in_dim) *in_dim = i_dim;
    return Error::success();
}

Error CudaRuntime::allocate_activation_buffers(int max_seq_len, int batch_size, int attn_q_max, int attn_k_max) {
    if (!activations_.empty() && 
        activations_[0].max_seq_len >= static_cast<size_t>(max_seq_len) &&
        activations_[0].batch_size >= static_cast<size_t>(batch_size) &&
        activations_[0].attn_q_max >= static_cast<size_t>(attn_q_max) &&
        activations_[0].attn_k_max >= static_cast<size_t>(attn_k_max)) {
        return Error::success();  // 已分配足够大的缓冲区
    }
    
    free_activation_buffers();
    
    size_t elem_size = dtype_size(weights_.dtype);
    int num_devices = device_map_.num_devices;
    activations_.resize(num_devices);
    
    for (int dev = 0; dev < num_devices; ++dev) {
        auto& act = activations_[dev];
        act.device_id = dev;
        act.max_seq_len = max_seq_len;
        act.batch_size = batch_size;
        act.attn_q_max = static_cast<size_t>(attn_q_max);
        act.attn_k_max = static_cast<size_t>(attn_k_max);
        
        size_t h = config_.hidden_size;
        size_t i = config_.intermediate_size;
        size_t nh = config_.num_heads;
        size_t nkv = config_.num_kv_heads;
        size_t hd = config_.head_dim;
        
        CUDA_CHECK(cudaSetDevice(dev));
        
        // 主要激活缓冲区
        size_t hidden_size = batch_size * max_seq_len * h * elem_size;
        CUDA_CHECK(cudaMalloc(&act.hidden_states, hidden_size));
        CUDA_CHECK(cudaMalloc(&act.norm_out, hidden_size));
        CUDA_CHECK(cudaMalloc(&act.last_hidden, batch_size * h * elem_size));
        
        // QKV 输出
        CUDA_CHECK(cudaMalloc(&act.q_proj_out, batch_size * max_seq_len * nh * hd * elem_size));
        CUDA_CHECK(cudaMalloc(&act.k_proj_out, batch_size * max_seq_len * nkv * hd * elem_size));
        CUDA_CHECK(cudaMalloc(&act.v_proj_out, batch_size * max_seq_len * nkv * hd * elem_size));
        
        // Attention 输出和分数
        CUDA_CHECK(cudaMalloc(&act.attn_out, batch_size * max_seq_len * nh * hd * elem_size));
        CUDA_CHECK(cudaMalloc(&act.attn_scores, batch_size * nh * attn_q_max * attn_k_max * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&act.attn_probs, batch_size * nh * attn_q_max * attn_k_max * elem_size));
        
        // MLP 缓冲区（gate/up 连续布局，便于 decode batched GEMM）
        const size_t mlp_proj_bytes = batch_size * max_seq_len * i * elem_size;
        CUDA_CHECK(cudaMalloc(&act.mlp_gate, mlp_proj_bytes * 2));
        act.mlp_up = static_cast<void*>(static_cast<char*>(act.mlp_gate) + mlp_proj_bytes);
        act.mlp_gate_up_packed = true;
        CUDA_CHECK(cudaMalloc(&act.mlp_down, hidden_size));
        
        // Logits（只需要最后一个 token）
        CUDA_CHECK(cudaMalloc(&act.logits, batch_size * config_.vocab_size * sizeof(float)));
        
        act.allocated = true;
    }
    
    return Error::success();
}

void CudaRuntime::free_activation_buffers() {
    for (auto& act : activations_) {
        if (act.allocated) {
            cudaSetDevice(act.device_id);
            cuda_free(act.hidden_states);
            cuda_free(act.norm_out);
            cuda_free(act.last_hidden);
            cuda_free(act.q_proj_out);
            cuda_free(act.k_proj_out);
            cuda_free(act.v_proj_out);
            cuda_free(act.attn_out);
            cuda_free(act.attn_scores);
            cuda_free(act.attn_probs);
            if (act.mlp_gate_up_packed) {
                cuda_free(act.mlp_gate);
                act.mlp_up = nullptr;
                act.mlp_gate_up_packed = false;
            } else {
                cuda_free(act.mlp_gate);
                cuda_free(act.mlp_up);
            }
            cuda_free(act.mlp_down);
            cuda_free(act.logits);
            act.allocated = false;
        }
    }
    activations_.clear();
}

Error CudaRuntime::allocate_kv_cache(Session& session) {
    auto& kv = session.kv_cache();
    
    for (int layer_idx = 0; layer_idx < kv.num_layers(); ++layer_idx) {
        int device_id = device_map_.layer_to_device[layer_idx];
        size_t layer_size = kv.layer_size_bytes() / 2;  // K 和 V 各一半
        
        void* k_data = nullptr;
        void* v_data = nullptr;
        
        Error err = cuda_malloc(&k_data, layer_size, device_id);
        if (err) return err;
        
        err = cuda_malloc(&v_data, layer_size, device_id);
        if (err) return err;
        
        // 清零
        err = cuda_memset(k_data, 0, layer_size, device_id);
        if (err) return err;
        err = cuda_memset(v_data, 0, layer_size, device_id);
        if (err) return err;
        
        kv.set_layer_data(layer_idx, k_data, v_data, device_id);
    }
    
    std::cout << "[CudaRuntime] Allocated KV cache: " 
              << format_bytes(kv.total_size_bytes()) << std::endl;
    
    return Error::success();
}

void CudaRuntime::free_kv_cache(Session& session) {
    auto& kv = session.kv_cache();
    
    for (int i = 0; i < kv.num_layers(); ++i) {
        auto& layer = kv.layer(i);
        if (layer.key_cache.data) {
            cudaSetDevice(layer.device_id);
            cuda_free(layer.key_cache.data);
            cuda_free(layer.value_cache.data);
            layer.key_cache.data = nullptr;
            layer.value_cache.data = nullptr;
        }
    }
}

void CudaRuntime::unload() {
    if (next_tokens_dev_) {
        int dev = 0;
        if (device_map_.num_devices > 0) dev = device_map_.lm_head_device;
        cudaSetDevice(dev);
        cuda_free(next_tokens_dev_);
        next_tokens_dev_ = nullptr;
        next_tokens_cap_ = 0;
    }

    // 释放激活缓冲区
    free_activation_buffers();
    
    // 释放权重
    if (weights_.embed_tokens) {
        cudaSetDevice(weights_.embed_device_id);
        cuda_free(weights_.embed_tokens);
    }
    if (weights_.lm_head && weights_.lm_head_owns_allocation) {
        cudaSetDevice(weights_.lm_head_device_id);
        cuda_free(weights_.lm_head);
    }
    if (weights_.final_norm) {
        cudaSetDevice(weights_.final_norm_device_id);
        cuda_free(weights_.final_norm);
    }
    
    weights_.embed_tokens = nullptr;
    weights_.lm_head = nullptr;
    weights_.final_norm = nullptr;
    weights_.lm_head_owns_allocation = false;
    
    for (auto& layer : weights_.layers) {
        if (layer.allocated) {
            cudaSetDevice(layer.device_id);
            cuda_free(layer.q_proj_weight);
            cuda_free(layer.k_proj_weight);
            cuda_free(layer.v_proj_weight);
            cuda_free(layer.o_proj_weight);
            cuda_free(layer.q_norm_weight);
            cuda_free(layer.k_norm_weight);
            if (layer.gate_up_proj_packed) {
                cuda_free(layer.gate_up_proj_weight);
            } else {
                cuda_free(layer.gate_proj_weight);
                cuda_free(layer.up_proj_weight);
            }
            cuda_free(layer.down_proj_weight);
            cuda_free(layer.input_layernorm_weight);
            cuda_free(layer.post_attention_layernorm_weight);
            layer.gate_proj_weight = nullptr;
            layer.up_proj_weight = nullptr;
            layer.gate_up_proj_weight = nullptr;
            layer.gate_up_proj_packed = false;
            layer.allocated = false;
        }
    }
    weights_.layers.clear();
    has_active_lora_adapter_ = false;
    active_lora_adapter_dir_.clear();
    active_lora_scale_ = 0.0f;
    
    // 销毁 streams 和 cuBLAS 句柄
    for (auto& stream : streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    streams_.clear();
    for (auto& stream : transfer_streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    transfer_streams_.clear();

    for (auto& ev : profile_events_) {
        if (ev.start) cudaEventDestroy(ev.start);
        if (ev.end) cudaEventDestroy(ev.end);
        ev.start = nullptr;
        ev.end = nullptr;
    }
    profile_events_.clear();
    destroy_stage_profiling_events_();
    cublas_handles_.clear();
    
    loaded_ = false;
}

void CudaRuntime::set_layer_profiling(bool enabled) {
    profile_layers_ = enabled;
}

std::vector<float> CudaRuntime::take_last_layer_profile_ms() {
    std::vector<float> out = last_layer_profile_ms_;
    std::fill(last_layer_profile_ms_.begin(), last_layer_profile_ms_.end(), 0.0f);
    return out;
}

void CudaRuntime::set_stage_profiling(bool enabled) {
    profile_stages_ = enabled;
}

CudaRuntime::StageProfileMs CudaRuntime::take_last_stage_profile_ms() {
    StageProfileMs out = last_stage_profile_ms_;
    last_stage_profile_ms_ = {};
    return out;
}

Error CudaRuntime::begin_stage_profile_() {
    if (!profile_stages_) {
        return Error::success();
    }
    ensure_stage_profiling_events_();
    last_stage_profile_ms_ = {};
    CUDA_CHECK(cudaSetDevice(total_events_.device_id));
    CUDA_CHECK(cudaEventRecord(total_events_.start, streams_[total_events_.device_id]));
    return Error::success();
}

Error CudaRuntime::finalize_stage_profile_(bool sync_all_devices,
                                           bool include_final_norm,
                                           bool include_lm_head) {
    if (!profile_stages_) {
        return Error::success();
    }
    if (sync_all_devices) {
        for (int dev = 0; dev < device_map_.num_devices; ++dev) {
            EMBER_RETURN_IF_ERROR(cuda_sync(dev));
        }
    }

    for (int l = 0; l < config_.num_layers; ++l) {
        const auto& ev = layer_stage_events_[static_cast<size_t>(l)];
        float ms = 0.0f;
        CUDA_CHECK(cudaSetDevice(ev.device_id));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.in_norm_start, ev.in_norm_end));
        last_stage_profile_ms_.rmsnorm_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.post_norm_start, ev.post_norm_end));
        last_stage_profile_ms_.rmsnorm_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.attn_start, ev.attn_end));
        last_stage_profile_ms_.attention_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.ffn_start, ev.ffn_end));
        last_stage_profile_ms_.ffn_ms += ms;
    }

    float emb_ms = 0.0f;
    CUDA_CHECK(cudaSetDevice(embedding_events_.device_id));
    CUDA_CHECK(cudaEventElapsedTime(&emb_ms, embedding_events_.start, embedding_events_.end));
    last_stage_profile_ms_.embedding_ms = emb_ms;

    if (include_final_norm) {
        float fn_ms = 0.0f;
        CUDA_CHECK(cudaSetDevice(final_norm_events_.device_id));
        CUDA_CHECK(cudaEventElapsedTime(&fn_ms, final_norm_events_.start, final_norm_events_.end));
        last_stage_profile_ms_.rmsnorm_ms += fn_ms;
    }

    if (include_lm_head) {
        float lm_ms = 0.0f;
        CUDA_CHECK(cudaSetDevice(lm_head_events_.device_id));
        CUDA_CHECK(cudaEventElapsedTime(&lm_ms, lm_head_events_.start, lm_head_events_.end));
        last_stage_profile_ms_.lm_head_ms = lm_ms;
    }

    CUDA_CHECK(cudaSetDevice(total_events_.device_id));
    CUDA_CHECK(cudaEventRecord(total_events_.end, streams_[total_events_.device_id]));
    CUDA_CHECK(cudaEventSynchronize(total_events_.end));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_events_.start, total_events_.end));
    last_stage_profile_ms_.total_ms = total_ms;
    return Error::success();
}

void CudaRuntime::add_stage_profile_h2d_ms_(double ms) {
    if (!profile_stages_) {
        return;
    }
    last_stage_profile_ms_.memcpy_h2d_ms += static_cast<float>(ms);
}

void CudaRuntime::add_stage_profile_d2h_ms_(double ms) {
    if (!profile_stages_) {
        return;
    }
    last_stage_profile_ms_.memcpy_d2h_ms += static_cast<float>(ms);
}

void CudaRuntime::destroy_stage_profiling_events_() {
    for (auto& ev : layer_stage_events_) {
        if (ev.device_id >= 0) cudaSetDevice(ev.device_id);
        if (ev.in_norm_start) cudaEventDestroy(ev.in_norm_start);
        if (ev.in_norm_end) cudaEventDestroy(ev.in_norm_end);
        if (ev.attn_start) cudaEventDestroy(ev.attn_start);
        if (ev.attn_end) cudaEventDestroy(ev.attn_end);
        if (ev.post_norm_start) cudaEventDestroy(ev.post_norm_start);
        if (ev.post_norm_end) cudaEventDestroy(ev.post_norm_end);
        if (ev.ffn_start) cudaEventDestroy(ev.ffn_start);
        if (ev.ffn_end) cudaEventDestroy(ev.ffn_end);
        ev = {};
    }
    layer_stage_events_.clear();

    auto destroy_simple = [](SimpleStageEvents& e) {
        if (e.device_id >= 0) cudaSetDevice(e.device_id);
        if (e.start) cudaEventDestroy(e.start);
        if (e.end) cudaEventDestroy(e.end);
        e = {};
    };
    destroy_simple(embedding_events_);
    destroy_simple(final_norm_events_);
    destroy_simple(lm_head_events_);
    destroy_simple(total_events_);
}

void CudaRuntime::ensure_simple_stage_events_(SimpleStageEvents& ev, int device_id) {
    if (ev.start && ev.end && ev.device_id == device_id) return;
    if (ev.start) cudaEventDestroy(ev.start);
    if (ev.end) cudaEventDestroy(ev.end);
    ev = {};
    ev.device_id = device_id;
    cudaSetDevice(device_id);
    cudaEventCreate(&ev.start);
    cudaEventCreate(&ev.end);
}

void CudaRuntime::ensure_layer_stage_events_(LayerStageEvents& ev, int device_id) {
    if (ev.in_norm_start && ev.in_norm_end && ev.attn_start && ev.attn_end &&
        ev.post_norm_start && ev.post_norm_end && ev.ffn_start && ev.ffn_end &&
        ev.device_id == device_id) {
        return;
    }
    if (ev.in_norm_start) cudaEventDestroy(ev.in_norm_start);
    if (ev.in_norm_end) cudaEventDestroy(ev.in_norm_end);
    if (ev.attn_start) cudaEventDestroy(ev.attn_start);
    if (ev.attn_end) cudaEventDestroy(ev.attn_end);
    if (ev.post_norm_start) cudaEventDestroy(ev.post_norm_start);
    if (ev.post_norm_end) cudaEventDestroy(ev.post_norm_end);
    if (ev.ffn_start) cudaEventDestroy(ev.ffn_start);
    if (ev.ffn_end) cudaEventDestroy(ev.ffn_end);
    ev = {};

    ev.device_id = device_id;
    cudaSetDevice(device_id);
    cudaEventCreate(&ev.in_norm_start);
    cudaEventCreate(&ev.in_norm_end);
    cudaEventCreate(&ev.attn_start);
    cudaEventCreate(&ev.attn_end);
    cudaEventCreate(&ev.post_norm_start);
    cudaEventCreate(&ev.post_norm_end);
    cudaEventCreate(&ev.ffn_start);
    cudaEventCreate(&ev.ffn_end);
}

void CudaRuntime::ensure_stage_profiling_events_() {
    if (!profile_stages_) return;
    if (layer_stage_events_.size() != static_cast<size_t>(config_.num_layers)) {
        layer_stage_events_.assign(static_cast<size_t>(config_.num_layers), {});
    }
    for (int l = 0; l < config_.num_layers; ++l) {
        ensure_layer_stage_events_(layer_stage_events_[static_cast<size_t>(l)], device_map_.layer_to_device[l]);
    }
    ensure_simple_stage_events_(embedding_events_, device_map_.embedding_device);
    ensure_simple_stage_events_(final_norm_events_, device_map_.lm_head_device);
    ensure_simple_stage_events_(lm_head_events_, device_map_.lm_head_device);
    ensure_simple_stage_events_(total_events_, device_map_.embedding_device);
}

Error CudaRuntime::prefill_chunked_pipeline(const std::vector<int>& tokens,
                                            Session& session,
                                            int chunk_len,
                                            bool overlap,
                                            std::vector<float>* out_logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (chunk_len <= 0) {
        return Error::invalid_argument("chunk_len must be > 0");
    }
    if (tokens.empty()) {
        return Error::invalid_argument("tokens is empty");
    }

    const int batch_size = 1;
    const int total_len = static_cast<int>(tokens.size());
    if (session.cur_pos() + total_len > session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }

    // B 阶段 baseline：仅支持 2-GPU、连续切分的 pipeline。
    if (device_map_.num_devices != 2) {
        return Error::invalid_argument("prefill_chunked_pipeline supports exactly 2 GPUs");
    }

    const int gpu0 = device_map_.embedding_device;
    const int gpu1 = device_map_.lm_head_device;
    if (gpu0 == gpu1) {
        return prefill(tokens, session);
    }

    // Find a single boundary: gpu0 layers first, then gpu1 layers.
    int boundary = -1;
    for (int i = 1; i < config_.num_layers; ++i) {
        if (device_map_.layer_to_device[i - 1] == gpu0 && device_map_.layer_to_device[i] == gpu1) {
            boundary = i;
            break;
        }
    }
    if (boundary <= 0 || boundary >= config_.num_layers) {
        return Error::invalid_argument("device_map must be a 2-stage contiguous split for pipeline prefill");
    }
    for (int i = 0; i < boundary; ++i) {
        if (device_map_.layer_to_device[i] != gpu0) {
            return Error::invalid_argument("non-contiguous split before boundary");
        }
    }
    for (int i = boundary; i < config_.num_layers; ++i) {
        if (device_map_.layer_to_device[i] != gpu1) {
            return Error::invalid_argument("non-contiguous split after boundary");
        }
    }

    const int max_ctx = session.runtime_config().max_ctx_len;
    // Token buffers only need to hold one chunk; attention workspace needs [seq_q=chunk_len, seq_k<=max_ctx].
    Error err = allocate_activation_buffers(chunk_len, batch_size, /*attn_q_max=*/chunk_len, /*attn_k_max=*/max_ctx);
    if (err) return err;

    EMBER_RETURN_IF_ERROR(begin_stage_profile_());

    auto accumulate_layer_stage_ms = [&](int layer_idx) -> Error {
        if (!profile_stages_) return Error::success();
        const auto& ev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        float ms = 0.0f;
        CUDA_CHECK(cudaSetDevice(ev.device_id));
        // Ensure the recorded end event has completed so elapsed-time queries are valid.
        CUDA_CHECK(cudaEventSynchronize(ev.ffn_end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.in_norm_start, ev.in_norm_end));
        last_stage_profile_ms_.rmsnorm_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.post_norm_start, ev.post_norm_end));
        last_stage_profile_ms_.rmsnorm_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.attn_start, ev.attn_end));
        last_stage_profile_ms_.attention_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.ffn_start, ev.ffn_end));
        last_stage_profile_ms_.ffn_ms += ms;
        return Error::success();
    };

    const DType compute_dtype = weights_.dtype;
    const size_t elem = dtype_size(compute_dtype);
    const size_t hidden = static_cast<size_t>(config_.hidden_size);
    const size_t max_chunk_bytes = static_cast<size_t>(chunk_len) * hidden * elem;

    int* d_input_ids = nullptr;
    void* stage0_out[2] = {nullptr, nullptr};
    void* gpu1_io[2] = {nullptr, nullptr};
    cudaEvent_t stage0_ready[2] = {nullptr, nullptr};
    cudaEvent_t xfer_done[2] = {nullptr, nullptr};
    cudaEvent_t stage1_done[2] = {nullptr, nullptr};

    auto cleanup = [&]() {
        cudaSetDevice(gpu0);
        if (d_input_ids) cudaFree(d_input_ids);
        if (stage0_out[0]) cudaFree(stage0_out[0]);
        if (stage0_out[1]) cudaFree(stage0_out[1]);
        if (stage0_ready[0]) cudaEventDestroy(stage0_ready[0]);
        if (stage0_ready[1]) cudaEventDestroy(stage0_ready[1]);
        if (xfer_done[0]) cudaEventDestroy(xfer_done[0]);
        if (xfer_done[1]) cudaEventDestroy(xfer_done[1]);

        cudaSetDevice(gpu1);
        if (gpu1_io[0]) cudaFree(gpu1_io[0]);
        if (gpu1_io[1]) cudaFree(gpu1_io[1]);
        if (stage1_done[0]) cudaEventDestroy(stage1_done[0]);
        if (stage1_done[1]) cudaEventDestroy(stage1_done[1]);
    };

    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaMalloc(&d_input_ids, static_cast<size_t>(chunk_len) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&stage0_out[0], max_chunk_bytes));
    CUDA_CHECK(cudaMalloc(&stage0_out[1], max_chunk_bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage0_ready[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage0_ready[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&xfer_done[0]));
    CUDA_CHECK(cudaEventCreate(&xfer_done[1]));

    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaMalloc(&gpu1_io[0], max_chunk_bytes));
    CUDA_CHECK(cudaMalloc(&gpu1_io[1], max_chunk_bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage1_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage1_done[1], cudaEventDisableTiming));

    const int num_chunks = (total_len + chunk_len - 1) / chunk_len;
    const int start_pos0 = session.cur_pos();

    struct TimedEvents {
        int device_id = -1;
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
    };
    std::vector<TimedEvents> copy_events;
    std::vector<TimedEvents> embed_events;

    auto destroy_copy_events = [&]() {
        for (auto& ce : copy_events) {
            if (ce.start) cudaEventDestroy(ce.start);
            if (ce.end) cudaEventDestroy(ce.end);
            ce = {};
        }
        copy_events.clear();
        for (auto& ee : embed_events) {
            if (ee.start) cudaEventDestroy(ee.start);
            if (ee.end) cudaEventDestroy(ee.end);
            ee = {};
        }
        embed_events.clear();
    };

    auto run_stage1 = [&](int chunk_index) -> Error {
        const int off = chunk_index * chunk_len;
        const int clen = std::min(chunk_len, total_len - off);
        const int slot = chunk_index & 1;

        // Ensure transfer is done (event recorded on gpu0).
        CUDA_CHECK(cudaSetDevice(gpu0));
        CUDA_CHECK(cudaEventSynchronize(xfer_done[slot]));

        CUDA_CHECK(cudaSetDevice(gpu1));
        void* saved_hidden = activations_[gpu1].hidden_states;
        activations_[gpu1].hidden_states = gpu1_io[slot];

        const int chunk_start_pos = start_pos0 + off;
        for (int layer = boundary; layer < config_.num_layers; ++layer) {
            const bool skip = (layer == boundary);
            Error e = forward_layer(layer, batch_size, clen, chunk_start_pos, session, /*skip_input_copy=*/skip);
            if (e) {
                activations_[gpu1].hidden_states = saved_hidden;
                return e;
            }
            Error pe = accumulate_layer_stage_ms(layer);
            if (pe) {
                activations_[gpu1].hidden_states = saved_hidden;
                return pe;
            }
        }

        // Mark this slot as free to reuse (recorded on gpu1 compute stream).
        CUDA_CHECK(cudaEventRecord(stage1_done[slot], streams_[gpu1]));
        activations_[gpu1].hidden_states = saved_hidden;
        return Error::success();
    };

    for (int i = 0; i < num_chunks; ++i) {
        const int off = i * chunk_len;
        const int clen = std::min(chunk_len, total_len - off);
        const int slot = i & 1;

        // Slot reuse guard: wait until stage1 has finished using this slot.
        if (i >= 2) {
            CUDA_CHECK(cudaSetDevice(gpu1));
            CUDA_CHECK(cudaEventSynchronize(stage1_done[slot]));
        }

        // --------------------
        // Stage 0 (GPU0): embedding + layers [0, boundary)
        // --------------------
        CUDA_CHECK(cudaSetDevice(gpu0));
        auto compute_stream0 = streams_[gpu0];
        auto transfer_stream0 = transfer_streams_[gpu0];

        CUDA_CHECK(cudaMemcpyAsync(d_input_ids, tokens.data() + off,
                                   static_cast<size_t>(clen) * sizeof(int),
                                   cudaMemcpyHostToDevice, compute_stream0));

        auto& act0 = activations_[gpu0];
        if (profile_stages_) {
            TimedEvents ee;
            ee.device_id = gpu0;
            CUDA_CHECK(cudaSetDevice(gpu0));
            CUDA_CHECK(cudaEventCreate(&ee.start));
            CUDA_CHECK(cudaEventCreate(&ee.end));
            CUDA_CHECK(cudaEventRecord(ee.start, compute_stream0));
            embed_events.push_back(ee);
        }
        if (weights_.dtype == DType::BF16) {
            kernels::embedding_lookup_bf16(
                static_cast<__nv_bfloat16*>(act0.hidden_states),
                static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
                d_input_ids,
                batch_size, clen, config_.hidden_size,
                compute_stream0
            );
        } else {
            kernels::embedding_lookup_f16(
                static_cast<half*>(act0.hidden_states),
                static_cast<const half*>(weights_.embed_tokens),
                d_input_ids,
                batch_size, clen, config_.hidden_size,
                compute_stream0
            );
        }
        if (profile_stages_) {
            auto& ee = embed_events.back();
            CUDA_CHECK(cudaEventRecord(ee.end, compute_stream0));
        }

        const int chunk_start_pos = start_pos0 + off;
        for (int layer = 0; layer < boundary; ++layer) {
            err = forward_layer(layer, batch_size, clen, chunk_start_pos, session, /*skip_input_copy=*/false);
            if (err) {
                destroy_copy_events();
                cleanup();
                return err;
            }
            Error pe = accumulate_layer_stage_ms(layer);
            if (pe) {
                destroy_copy_events();
                cleanup();
                return pe;
            }
        }

        // Snapshot stage0 output so GPU0 can continue computing next chunk while transfer reads this buffer.
        const size_t bytes = static_cast<size_t>(clen) * hidden * elem;
        CUDA_CHECK(cudaMemcpyAsync(stage0_out[slot], act0.hidden_states, bytes,
                                   cudaMemcpyDeviceToDevice, compute_stream0));
        CUDA_CHECK(cudaEventRecord(stage0_ready[slot], compute_stream0));

        // --------------------
        // Transfer (GPU0): stage0_out -> gpu1_io
        // --------------------
        if (overlap) {
            CUDA_CHECK(cudaStreamWaitEvent(transfer_stream0, stage0_ready[slot], 0));
            if (profile_stages_) {
                TimedEvents ce;
                ce.device_id = gpu0;
                CUDA_CHECK(cudaSetDevice(gpu0));
                CUDA_CHECK(cudaEventCreate(&ce.start));
                CUDA_CHECK(cudaEventCreate(&ce.end));
                CUDA_CHECK(cudaEventRecord(ce.start, transfer_stream0));
                copy_events.push_back(ce);
            }
            cudaError_t cperr = cudaMemcpyPeerAsync(gpu1_io[slot], gpu1, stage0_out[slot], gpu0, bytes, transfer_stream0);
            if (cperr != cudaSuccess) {
                // Fall back to staging (synchronous) on failure.
                cudaGetLastError();
                CUDA_CHECK(cudaStreamSynchronize(transfer_stream0));
                Error se = copy_bytes_peer_or_staged(gpu1_io[slot], gpu1, stage0_out[slot], gpu0, bytes);
                if (se) {
                    destroy_copy_events();
                    cleanup();
                    return se;
                }
            }
            if (profile_stages_) {
                auto& ce = copy_events.back();
                CUDA_CHECK(cudaEventRecord(ce.end, transfer_stream0));
            }
            CUDA_CHECK(cudaEventRecord(xfer_done[slot], transfer_stream0));
        } else {
            // Baseline: block on stage0 completion, then do synchronous peer copy.
            CUDA_CHECK(cudaEventSynchronize(stage0_ready[slot]));
            auto t0 = std::chrono::high_resolution_clock::now();
            err = copy_bytes_peer_or_staged(gpu1_io[slot], gpu1, stage0_out[slot], gpu0, bytes);
            if (err) {
                destroy_copy_events();
                cleanup();
                return err;
            }
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            CUDA_CHECK(cudaEventRecord(xfer_done[slot], compute_stream0));
        }

        // --------------------
        // Stage 1 (GPU1): consume previous chunk if available.
        // --------------------
        if (i - 1 >= 0) {
            Error e = run_stage1(i - 1);
            if (e) {
                destroy_copy_events();
                cleanup();
                return e;
            }
        }
    }

    // Consume the last chunk on stage 1.
    if (num_chunks > 0) {
        err = run_stage1(num_chunks - 1);
        if (err) {
            destroy_copy_events();
            cleanup();
            return err;
        }
    }

    if (out_logits) {
        // Compute logits for the last prompt token from the last chunk output on GPU1.
        const int last_chunk_index = num_chunks - 1;
        const int last_off = last_chunk_index * chunk_len;
        const int last_clen = std::min(chunk_len, total_len - last_off);
        const int slot = last_chunk_index & 1;

        const size_t bytes_per_token = hidden * elem;
        const char* base = static_cast<const char*>(gpu1_io[slot]);
        const void* last_hidden_ptr = static_cast<const void*>(base + static_cast<size_t>(last_clen - 1) * bytes_per_token);

        CUDA_CHECK(cudaSetDevice(gpu1));
        auto& act1 = activations_[gpu1];
        auto stream1 = streams_[gpu1];

        if (compute_dtype == DType::BF16) {
            kernels::rms_norm_bf16(
                static_cast<__nv_bfloat16*>(act1.norm_out),
                static_cast<const __nv_bfloat16*>(last_hidden_ptr),
                static_cast<const __nv_bfloat16*>(weights_.final_norm),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                config_.rms_norm_eps,
                stream1
            );
            kernels::copy_last_hidden_bf16(
                static_cast<__nv_bfloat16*>(act1.last_hidden),
                static_cast<const __nv_bfloat16*>(act1.norm_out),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                stream1
            );
        } else {
            kernels::rms_norm_f16(
                static_cast<half*>(act1.norm_out),
                static_cast<const half*>(last_hidden_ptr),
                static_cast<const half*>(weights_.final_norm),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                config_.rms_norm_eps,
                stream1
            );
            kernels::copy_last_hidden_f16(
                static_cast<half*>(act1.last_hidden),
                static_cast<const half*>(act1.norm_out),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                stream1
            );
        }

        // Temporarily run lm_head with batch_size=1 using last_hidden.
        // forward_lm_head will read act1.last_hidden for batch>0 after we fix it.
        err = forward_lm_head(/*batch_size=*/1, /*seq_len=*/1);
        if (err) {
            destroy_copy_events();
            cleanup();
            return err;
        }
        cuda_sync(gpu1);

        out_logits->resize(static_cast<size_t>(config_.vocab_size));
        CUDA_CHECK(cudaMemcpy(out_logits->data(), act1.logits,
                              static_cast<size_t>(config_.vocab_size) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // Sync both devices to ensure KV cache writes are visible before decode.
    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    if (profile_stages_) {
        // Sum per-chunk embedding time.
        float emb_sum = 0.0f;
        for (auto& ee : embed_events) {
            if (!ee.start || !ee.end) continue;
            float ems = 0.0f;
            CUDA_CHECK(cudaSetDevice(ee.device_id));
            CUDA_CHECK(cudaEventElapsedTime(&ems, ee.start, ee.end));
            emb_sum += ems;
        }
        last_stage_profile_ms_.embedding_ms = emb_sum;

        // Sum async copy times.
        for (auto& ce : copy_events) {
            if (!ce.start || !ce.end) continue;
            float cms = 0.0f;
            CUDA_CHECK(cudaSetDevice(ce.device_id));
            CUDA_CHECK(cudaEventElapsedTime(&cms, ce.start, ce.end));
            last_stage_profile_ms_.p2p_ms += cms;
        }

        CUDA_CHECK(cudaSetDevice(total_events_.device_id));
        CUDA_CHECK(cudaEventRecord(total_events_.end, streams_[total_events_.device_id]));
        CUDA_CHECK(cudaEventSynchronize(total_events_.end));
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_events_.start, total_events_.end));
        last_stage_profile_ms_.total_ms = total_ms;
    }

    destroy_copy_events();
    session.set_cur_pos(0, start_pos0 + total_len);
    cleanup();
    return Error::success();
}

Error CudaRuntime::prefill_batch_flat(const std::vector<int>& input_ids_flat,
                                      int seq_len,
                                      Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    const int batch_size = session.runtime_config().batch_size;
    if (batch_size <= 0) {
        return Error::invalid_argument("runtime_config.batch_size must be > 0");
    }
    if (seq_len <= 0) {
        return Error::invalid_argument("seq_len must be > 0");
    }
    if (static_cast<int>(input_ids_flat.size()) != batch_size * seq_len) {
        return Error::invalid_argument("input_ids_flat size must equal batch_size * seq_len");
    }
    if (seq_len > session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }

    const int max_ctx = session.runtime_config().max_ctx_len;
    Error err = allocate_activation_buffers(seq_len, batch_size, /*attn_q_max=*/seq_len, /*attn_k_max=*/max_ctx);
    if (err) return err;

    int embed_device = device_map_.embedding_device;
    CUDA_CHECK(cudaSetDevice(embed_device));

    int* d_input_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_ids, static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input_ids, input_ids_flat.data(),
                          static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) * sizeof(int),
                          cudaMemcpyHostToDevice));

    auto& act = activations_[embed_device];
    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    cudaFree(d_input_ids);

    const int start_pos = session.cur_pos();
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }

    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    session.set_cur_pos(start_pos + seq_len);
    return Error::success();
}

Error CudaRuntime::prefill_into_slot(const std::vector<int>& tokens,
                                     int slot,
                                     Session& session,
                                     std::vector<float>* out_logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (tokens.empty()) {
        return Error::invalid_argument("tokens is empty");
    }
    const int batch_cap = session.runtime_config().batch_size;
    if (slot < 0 || slot >= batch_cap) {
        return Error::invalid_argument("slot out of range for runtime_config.batch_size");
    }
    const int seq_len = static_cast<int>(tokens.size());
    const int max_ctx = session.runtime_config().max_ctx_len;

    int start_pos = session.cur_pos(slot);
    if (start_pos < 0) {
        start_pos = 0;
        session.set_cur_pos(slot, 0);
    }
    if (start_pos + seq_len > max_ctx) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }

    auto guard_r = ember::KVCacheSlotGuard::create(session, slot);
    if (!guard_r.ok()) return guard_r.error();
    auto guard = std::move(guard_r.value());

    Error err = allocate_activation_buffers(seq_len, /*batch_size=*/1, /*attn_q_max=*/seq_len, /*attn_k_max=*/max_ctx);
    if (err) {
        return err;
    }

    int embed_device = device_map_.embedding_device;
    CUDA_CHECK(cudaSetDevice(embed_device));

    int* d_input_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input_ids, static_cast<size_t>(seq_len) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input_ids, tokens.data(),
                          static_cast<size_t>(seq_len) * sizeof(int),
                          cudaMemcpyHostToDevice));

    auto& act = activations_[embed_device];
    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_ids,
            /*batch_size=*/1,
            seq_len,
            config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_ids,
            /*batch_size=*/1,
            seq_len,
            config_.hidden_size,
            streams_[embed_device]
        );
    }
    cudaFree(d_input_ids);

    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx, /*batch_size=*/1, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) {
            return err;
        }
    }

    if (out_logits) {
        err = forward_final_norm(/*batch_size=*/1, seq_len, session);
        if (err) {
            return err;
        }
        err = forward_lm_head(/*batch_size=*/1, seq_len);
        if (err) {
            return err;
        }
        const int lm_device = device_map_.lm_head_device;
        cuda_sync(lm_device);
        out_logits->resize(static_cast<size_t>(config_.vocab_size));
        CUDA_CHECK(cudaMemcpy(out_logits->data(), activations_[lm_device].logits,
                              out_logits->size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    session.set_cur_pos(slot, start_pos + seq_len);
    return Error::success();
}

Error CudaRuntime::prefill_into_slot_pipeline(const std::vector<int>& tokens,
                                              int slot,
                                              Session& session,
                                              int chunk_len,
                                              bool overlap,
                                              std::vector<float>* out_logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (tokens.empty()) {
        return Error::invalid_argument("tokens is empty");
    }
    if (chunk_len <= 0) {
        return Error::invalid_argument("chunk_len must be > 0");
    }
    const int batch_cap = session.runtime_config().batch_size;
    if (slot < 0 || slot >= batch_cap) {
        return Error::invalid_argument("slot out of range for runtime_config.batch_size");
    }
    if (device_map_.num_devices != 2) {
        return Error::invalid_argument("prefill_into_slot_pipeline requires exactly 2 GPUs");
    }

    int start_pos = session.cur_pos(slot);
    if (start_pos < 0) {
        start_pos = 0;
        session.set_cur_pos(slot, 0);
    }
    const int total_len = static_cast<int>(tokens.size());
    const int max_ctx = session.runtime_config().max_ctx_len;
    if (start_pos + total_len > max_ctx) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }

    auto guard_r = ember::KVCacheSlotGuard::create(session, slot);
    if (!guard_r.ok()) return guard_r.error();
    auto guard = std::move(guard_r.value());

    // prefill_chunked_pipeline uses session.cur_pos() (slot0). Temporarily mirror this slot's
    // position into slot0, then after success copy the updated pos back into this slot.
    if (slot != 0) {
        auto pos_guard = ember::CurPosGuard::set(session, /*slot=*/0, /*new_pos=*/start_pos);
        Error err = prefill_chunked_pipeline(tokens, session, chunk_len, overlap, out_logits);
        if (err) {
            return err;  // pos_guard restores slot0 automatically
        }
        const int new_pos0 = session.cur_pos(0);
        session.set_cur_pos(slot, new_pos0);
        return Error::success();
    }

    // slot0: no swap needed; prefill_chunked_pipeline already updates slot0.
    return prefill_chunked_pipeline(tokens, session, chunk_len, overlap, out_logits);
}

Error CudaRuntime::decode_single_forward_to_lm_head_(int last_token, Session& session) {
    const int batch_size = 1;
    const int seq_len = 1;
    const int start_pos = session.cur_pos();
    if (start_pos >= session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full");
    }

    const int max_ctx = session.runtime_config().max_ctx_len;
    Error err = allocate_activation_buffers(/*max_seq_len=*/1, batch_size, /*attn_q_max=*/1, /*attn_k_max=*/max_ctx);
    if (err) return err;

    int embed_device = device_map_.embedding_device;
    auto& act = activations_[embed_device];

    EMBER_RETURN_IF_ERROR(begin_stage_profile_());

    int* d_input_id = nullptr;
    CUDA_CHECK(cudaSetDevice(embed_device));
    CUDA_CHECK(cudaMalloc(&d_input_id, sizeof(int)));
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input_id, &last_token, sizeof(int), cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_h2d_ms_(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());

    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.start, streams_[embed_device]));
    }
    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_id,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_id,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.end, streams_[embed_device]));
    }
    cudaFree(d_input_id);

    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }

    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;
    return Error::success();
}

Error CudaRuntime::decode_to_device(int last_token, Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    EMBER_RETURN_IF_ERROR(decode_single_forward_to_lm_head_(last_token, session));
    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/true,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    session.advance(1);
    return Error::success();
}

Error CudaRuntime::decode_batch_to_device(const std::vector<int>& last_tokens, Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    const int batch_size = session.runtime_config().batch_size;
    const int seq_len = 1;
    if (batch_size <= 0) {
        return Error::invalid_argument("runtime_config.batch_size must be > 0");
    }
    if (static_cast<int>(last_tokens.size()) != batch_size) {
        return Error::invalid_argument("last_tokens size must equal runtime_config.batch_size");
    }
    const int max_ctx = session.runtime_config().max_ctx_len;

    std::vector<int> start_pos_by_batch(static_cast<size_t>(batch_size), -1);
    bool any_inactive = false;
    bool all_same = true;
    int first_active_pos = -1;
    for (int b = 0; b < batch_size; ++b) {
        const int sp = session.cur_pos(b);
        start_pos_by_batch[static_cast<size_t>(b)] = sp;
        if (sp < 0) {
            any_inactive = true;
            continue;
        }
        if (sp >= max_ctx) {
            return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full");
        }
        if (first_active_pos < 0) first_active_pos = sp;
        else if (sp != first_active_pos) all_same = false;
    }
    if (first_active_pos < 0) {
        return Error::invalid_argument("decode_batch_to_device: no active slots");
    }
    const bool use_varpos = any_inactive || !all_same;

    Error err = allocate_activation_buffers(/*max_seq_len=*/1, batch_size, /*attn_q_max=*/1, /*attn_k_max=*/max_ctx);
    if (err) return err;

    int embed_device = device_map_.embedding_device;
    auto& act = activations_[embed_device];

    int* d_input_ids = nullptr;
    CUDA_CHECK(cudaSetDevice(embed_device));
    CUDA_CHECK(cudaMalloc(&d_input_ids, static_cast<size_t>(batch_size) * sizeof(int)));
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input_ids, last_tokens.data(),
                          static_cast<size_t>(batch_size) * sizeof(int),
                          cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_h2d_ms_(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());

    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    cudaFree(d_input_ids);

    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx,
                            batch_size,
                            seq_len,
                            use_varpos ? 0 : first_active_pos,
                            session,
                            /*skip_input_copy=*/false,
                            use_varpos ? start_pos_by_batch.data() : nullptr);
        if (err) return err;
    }

    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;

    if (!use_varpos) {
        session.advance(1);
    } else {
        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[static_cast<size_t>(b)];
            if (sp >= 0) session.set_cur_pos(b, sp + 1);
        }
    }
    return Error::success();
}

Error CudaRuntime::prefill(const std::vector<int>& tokens, Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    
    int batch_size = 1;
    int seq_len = static_cast<int>(tokens.size());
    
    if (seq_len > session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }
    
    // 确保激活缓冲区已分配
    const int max_ctx = session.runtime_config().max_ctx_len;
    Error err = allocate_activation_buffers(seq_len, batch_size, /*attn_q_max=*/seq_len, /*attn_k_max=*/max_ctx);
    if (err) return err;

    if (profile_layers_) {
        last_layer_profile_ms_.assign(static_cast<size_t>(config_.num_layers), 0.0f);
    }
    EMBER_RETURN_IF_ERROR(begin_stage_profile_());
    
    // 拷贝 input_ids 到 GPU
    input_ids_cpu_ = tokens;
    int* d_input_ids = nullptr;
    int embed_device = device_map_.embedding_device;
    
    CUDA_CHECK(cudaSetDevice(embed_device));
    CUDA_CHECK(cudaMalloc(&d_input_ids, seq_len * sizeof(int)));
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input_ids, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_h2d_ms_(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());
    
    // Embedding lookup
    auto& act = activations_[embed_device];
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.start, streams_[embed_device]));
    }
    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.end, streams_[embed_device]));
    }
    
    cudaFree(d_input_ids);
    
    // 逐层前向
    int start_pos = session.cur_pos();
    
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }
    
    // 同步所有设备
    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/false,
                                                  /*include_lm_head=*/false));
    
    // 更新位置
    session.set_cur_pos(start_pos + seq_len);
    
    return Error::success();
}

// 带 logits 返回的 prefill (用于立即采样)
Error CudaRuntime::prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) {
    Error err = prefill(tokens, session);
    if (err) return err;
    
    // Final norm 和 LM head 来计算最后一个位置的 logits
    int batch_size = 1;
    int seq_len = static_cast<int>(tokens.size());
    
    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;
    
    // 拷贝 logits
    int lm_device = device_map_.lm_head_device;
    cuda_sync(lm_device);
    
    logits.resize(config_.vocab_size);
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits.data(), activations_[lm_device].logits, 
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());
    
    return Error::success();
}

Error CudaRuntime::decode(int last_token, Session& session, std::vector<float>& logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }

    if (profile_layers_) {
        last_layer_profile_ms_.assign(static_cast<size_t>(config_.num_layers), 0.0f);
    }
    EMBER_RETURN_IF_ERROR(decode_single_forward_to_lm_head_(last_token, session));
    
    // 同步并拷贝 logits 回 CPU
    int lm_device = device_map_.lm_head_device;
    if (profile_stages_) {
        for (int dev = 0; dev < device_map_.num_devices; ++dev) {
            cuda_sync(dev);
        }
    } else {
    cuda_sync(lm_device);
    }
    
    logits.resize(config_.vocab_size);
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits.data(), activations_[lm_device].logits, 
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    
    // 更新位置
    session.advance(1);
    
    return Error::success();
}

Error CudaRuntime::forward_layer(int layer_idx,
                                 int batch_size,
                                 int seq_len,
                                 int start_pos,
                                 Session& session,
                                 bool skip_input_copy,
                                 const int* start_pos_by_batch) {
    int device_id = device_map_.layer_to_device[layer_idx];
    auto& act = activations_[device_id];
    auto& layer = weights_.layers[layer_idx];
    auto stream = streams_[device_id];
    auto& cublas = cublas_handles_[device_id];
    
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasSetStream(cublas.get(), stream);
    
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    size_t hidden_size = config_.hidden_size;
    size_t num_heads = config_.num_heads;
    size_t num_kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim;
    size_t intermediate_size = config_.intermediate_size;
    size_t elem_size = dtype_size(compute_dtype);
    
    // 如果上一层在不同设备，需要拷贝 hidden_states
    if (layer_idx > 0) {
        int prev_device = device_map_.layer_to_device[layer_idx - 1];
        if (prev_device != device_id) {
            if (!skip_input_copy) {
            size_t size = batch_size * seq_len * hidden_size * elem_size;
            auto t0 = std::chrono::high_resolution_clock::now();
            Error err = copy_bytes_peer_or_staged(act.hidden_states, device_id,
                                                  activations_[prev_device].hidden_states, prev_device,
                                                  size);
            if (err) return err;
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            }
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.start, stream));
    }
    
    // =====================================================================
    // Input LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.attn_start, stream));
    }
    
    // =====================================================================
    // QKV Projection: norm_out @ W_q/k/v -> q/k/v_proj_out
    // 权重布局: [out_features, in_features] (row-major, Qwen3 safetensors格式)
    // 输入: [batch*seq, hidden_size]
    // 输出: [batch*seq, out_features]
    // =====================================================================
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    
    int M = batch_size * seq_len;  // 批次*序列长度
    
    // Q projection: [M, hidden] @ [hidden, num_heads*head_dim]^T = [M, num_heads*head_dim]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,  // W^T @ X
        num_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.q_proj_weight, cuda_dtype, hidden_size,  // W: [num_heads*head_dim, hidden] -> W^T
        act.norm_out, cuda_dtype, hidden_size,         // X: [M, hidden]
        &beta_zero,
        act.q_proj_out, cuda_dtype, num_heads * head_dim,  // Y: [M, num_heads*head_dim]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // K projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.k_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.k_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // V projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    }
    
    // =====================================================================
    // Apply RoPE to Q and K
    // =====================================================================
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::apply_rope_bf16(
                static_cast<__nv_bfloat16*>(act.q_proj_out),
                static_cast<__nv_bfloat16*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        } else {
            kernels::apply_rope_f16(
                static_cast<half*>(act.q_proj_out),
                static_cast<half*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        }
    }
    
    // =====================================================================
    // Update KV Cache
    // =====================================================================
    auto& kv_cache = session.kv_cache();
    auto& layer_kv = kv_cache.layer(layer_idx);
    
    const int max_seq = session.runtime_config().max_ctx_len;
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::update_kv_cache_bf16(
                static_cast<__nv_bfloat16*>(layer_kv.key_cache.data),
                static_cast<__nv_bfloat16*>(layer_kv.value_cache.data),
                static_cast<const __nv_bfloat16*>(act.k_proj_out),
                static_cast<const __nv_bfloat16*>(act.v_proj_out),
                batch_size, seq_len, num_kv_heads, head_dim,
                start_pos, max_seq,
                stream
            );
        } else {
            kernels::update_kv_cache_f16(
                static_cast<half*>(layer_kv.key_cache.data),
                static_cast<half*>(layer_kv.value_cache.data),
                static_cast<const half*>(act.k_proj_out),
                static_cast<const half*>(act.v_proj_out),
                batch_size, seq_len, num_kv_heads, head_dim,
                start_pos, max_seq,
                stream
            );
        }
    } else {
        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[b];
            if (sp < 0) continue;
            if (sp + seq_len > max_seq) {
                return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full (varpos batch)");
            }

            const size_t q_off = static_cast<size_t>(b) * static_cast<size_t>(seq_len) *
                                 static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim);
            const size_t kv_off = static_cast<size_t>(b) * static_cast<size_t>(seq_len) *
                                  static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);
            const size_t cache_off = static_cast<size_t>(b) * static_cast<size_t>(num_kv_heads) *
                                     static_cast<size_t>(max_seq) * static_cast<size_t>(head_dim);

            if (compute_dtype == DType::BF16) {
                auto* q_ptr = static_cast<__nv_bfloat16*>(act.q_proj_out) + q_off;
                auto* k_ptr = static_cast<__nv_bfloat16*>(act.k_proj_out) + kv_off;
                auto* v_ptr = static_cast<__nv_bfloat16*>(act.v_proj_out) + kv_off;
                auto* k_cache_ptr = static_cast<__nv_bfloat16*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<__nv_bfloat16*>(layer_kv.value_cache.data) + cache_off;

                kernels::apply_rope_bf16(
                    q_ptr,
                    k_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_bf16(
                    k_cache_ptr,
                    v_cache_ptr,
                    k_ptr,
                    v_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    sp,
                    max_seq,
                    stream
                );
            } else {
                auto* q_ptr = static_cast<half*>(act.q_proj_out) + q_off;
                auto* k_ptr = static_cast<half*>(act.k_proj_out) + kv_off;
                auto* v_ptr = static_cast<half*>(act.v_proj_out) + kv_off;
                auto* k_cache_ptr = static_cast<half*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<half*>(layer_kv.value_cache.data) + cache_off;

                kernels::apply_rope_f16(
                    q_ptr,
                    k_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_f16(
                    k_cache_ptr,
                    v_cache_ptr,
                    k_ptr,
                    v_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    sp,
                    max_seq,
                    stream
                );
            }
        }
    }
    
    // =====================================================================
    // Attention: Q @ K^T / sqrt(d) -> Softmax -> @ V
    // =====================================================================
    int seq_k = start_pos + seq_len;  // KV cache 中的有效长度
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::attention_bf16(
                static_cast<__nv_bfloat16*>(act.attn_out),
                static_cast<const __nv_bfloat16*>(act.q_proj_out),
                static_cast<const __nv_bfloat16*>(layer_kv.key_cache.data),
                static_cast<const __nv_bfloat16*>(layer_kv.value_cache.data),
                static_cast<float*>(act.attn_scores),
                static_cast<__nv_bfloat16*>(act.attn_probs),
                batch_size, seq_len, seq_k, max_seq, start_pos,
                num_heads, num_kv_heads, head_dim,
                scale, cublas.get(), stream
            );
        } else {
            kernels::attention_f16(
                static_cast<half*>(act.attn_out),
                static_cast<const half*>(act.q_proj_out),
                static_cast<const half*>(layer_kv.key_cache.data),
                static_cast<const half*>(layer_kv.value_cache.data),
                static_cast<float*>(act.attn_scores),
                static_cast<half*>(act.attn_probs),
                batch_size, seq_len, seq_k, max_seq, start_pos,
                num_heads, num_kv_heads, head_dim,
                scale, cublas.get(), stream
            );
        }
    } else {
        const size_t attn_out_batch_stride =
            static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim);
        const size_t cache_batch_stride =
            static_cast<size_t>(num_kv_heads) * static_cast<size_t>(max_seq) * static_cast<size_t>(head_dim);
        const size_t score_batch_stride =
            static_cast<size_t>(num_heads) * act.attn_q_max * act.attn_k_max;

        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[b];

            const size_t attn_off = static_cast<size_t>(b) * attn_out_batch_stride;
            const size_t q_off =
                static_cast<size_t>(b) * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) *
                static_cast<size_t>(head_dim);
            const size_t cache_off = static_cast<size_t>(b) * cache_batch_stride;
            const size_t score_off = static_cast<size_t>(b) * score_batch_stride;

            if (sp < 0) {
                CUDA_CHECK(cudaMemsetAsync(
                    static_cast<char*>(act.attn_out) + attn_off * elem_size,
                    0,
                    attn_out_batch_stride * elem_size,
                    stream));
                continue;
            }

            const int seq_k_b = sp + seq_len;
            if (seq_k_b <= 0) {
                CUDA_CHECK(cudaMemsetAsync(
                    static_cast<char*>(act.attn_out) + attn_off * elem_size,
                    0,
                    attn_out_batch_stride * elem_size,
                    stream));
                continue;
            }

            if (compute_dtype == DType::BF16) {
                auto* out_ptr = static_cast<__nv_bfloat16*>(act.attn_out) + attn_off;
                auto* q_ptr = static_cast<const __nv_bfloat16*>(act.q_proj_out) + q_off;
                auto* k_cache_ptr = static_cast<const __nv_bfloat16*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<const __nv_bfloat16*>(layer_kv.value_cache.data) + cache_off;
                auto* scores_ptr = static_cast<float*>(act.attn_scores) + score_off;
                auto* probs_ptr = static_cast<__nv_bfloat16*>(act.attn_probs) + score_off;

                kernels::attention_bf16(
                    out_ptr,
                    q_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    scores_ptr,
                    probs_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    seq_k_b,
                    max_seq,
                    sp,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    cublas.get(),
                    stream
                );
            } else {
                auto* out_ptr = static_cast<half*>(act.attn_out) + attn_off;
                auto* q_ptr = static_cast<const half*>(act.q_proj_out) + q_off;
                auto* k_cache_ptr = static_cast<const half*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<const half*>(layer_kv.value_cache.data) + cache_off;
                auto* scores_ptr = static_cast<float*>(act.attn_scores) + score_off;
                auto* probs_ptr = static_cast<half*>(act.attn_probs) + score_off;

                kernels::attention_f16(
                    out_ptr,
                    q_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    scores_ptr,
                    probs_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    seq_k_b,
                    max_seq,
                    sp,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    cublas.get(),
                    stream
                );
            }
        }
    }
    
    // =====================================================================
    // O Projection: attn_out @ W_o -> hidden_states
    // attn_out: [M, num_heads*head_dim]
    // W_o: [hidden_size, num_heads*head_dim]
    // output: [M, hidden_size]
    // =====================================================================
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, M, num_heads * head_dim,
        &alpha_one,
        layer.o_proj_weight, cuda_dtype, num_heads * head_dim,
        act.attn_out, cuda_dtype, num_heads * head_dim,
        &beta_zero,
        act.mlp_down, cuda_dtype, hidden_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_attn_out",
                                     device_id, act.mlp_down,
                                     seq_len, hidden_size, compute_dtype);
        if (err) return err;
    }
    
    // =====================================================================
    // Residual Connection (Attention)
    // =====================================================================
    if (compute_dtype == DType::BF16) {
        kernels::elementwise_add_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(act.mlp_down),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            batch_size * seq_len * hidden_size,
            stream
        );
    } else {
        kernels::elementwise_add_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(act.mlp_down),
            static_cast<const half*>(act.hidden_states),
            batch_size * seq_len * hidden_size,
            stream
        );
    }

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_attn_residual",
                                     device_id, act.hidden_states,
                                     seq_len, hidden_size, compute_dtype);
        if (err) return err;
    }
    
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.attn_end, stream));
    }
    
    // =====================================================================
    // Post-Attention LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.post_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.post_attention_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.post_attention_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.post_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.ffn_start, stream));
    }

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_post_attn_norm",
                                     device_id, act.norm_out,
                                     seq_len, hidden_size, compute_dtype);
        if (err) return err;
    }
    
    // =====================================================================
    // MLP: SwiGLU
    // gate = SiLU(norm_out @ W_gate)
    // up = norm_out @ W_up
    // down = (gate * up) @ W_down
    // =====================================================================
    
    if (seq_len == 1 && layer.gate_up_proj_packed && act.mlp_gate_up_packed) {
        const int64_t stride_a = static_cast<int64_t>(intermediate_size) * hidden_size;
        const int64_t stride_c = static_cast<int64_t>(intermediate_size) *
                                 static_cast<int64_t>(act.batch_size * act.max_seq_len);
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, M, hidden_size,
            &alpha_one,
            layer.gate_proj_weight, cuda_dtype, hidden_size, stride_a,
            act.norm_out, cuda_dtype, hidden_size, 0,
            &beta_zero,
            act.mlp_gate, cuda_dtype, intermediate_size, stride_c,
            2,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else {
        // Gate projection: [M, hidden] @ [intermediate, hidden]^T = [M, intermediate]
        CUBLAS_CHECK(cublasGemmEx(
            cublas.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, M, hidden_size,
            &alpha_one,
            layer.gate_proj_weight, cuda_dtype, hidden_size,
            act.norm_out, cuda_dtype, hidden_size,
            &beta_zero,
            act.mlp_gate, cuda_dtype, intermediate_size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // Up projection
        CUBLAS_CHECK(cublasGemmEx(
            cublas.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, M, hidden_size,
            &alpha_one,
            layer.up_proj_weight, cuda_dtype, hidden_size,
            act.norm_out, cuda_dtype, hidden_size,
            &beta_zero,
            act.mlp_up, cuda_dtype, intermediate_size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                  "layer_" + std::to_string(layer_idx) + "_mlp_gate_pre",
                                  device_id, act.mlp_gate,
                                  seq_len, intermediate_size, compute_dtype);
        if (err) return err;
    }
    
    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                  "layer_" + std::to_string(layer_idx) + "_mlp_up",
                                  device_id, act.mlp_up,
                                  seq_len, intermediate_size, compute_dtype);
        if (err) return err;
    }
    
    const bool dump_mlp_intermediates =
        session.runtime_config().check_correctness && session.runtime_config().dump_layer == layer_idx;
    if (dump_mlp_intermediates) {
        // Debug path: keep intermediate tensors materialized for compare_hidden.py.
        if (compute_dtype == DType::BF16) {
            kernels::silu_bf16(
                static_cast<__nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_gate),
                batch_size * seq_len * intermediate_size,
                stream
            );
        } else {
            kernels::silu_f16(
                static_cast<half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_gate),
                batch_size * seq_len * intermediate_size,
                stream
            );
        }

        Error err = dump_last_row(session.runtime_config().dump_dir,
                                  "layer_" + std::to_string(layer_idx) + "_mlp_gate_act",
                                  device_id, act.mlp_gate,
                                  seq_len, intermediate_size, compute_dtype);
        if (err) return err;

        if (compute_dtype == DType::BF16) {
            kernels::elementwise_mul_bf16(
                static_cast<__nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        } else {
            kernels::elementwise_mul_f16(
                static_cast<half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        }

        err = dump_last_row(session.runtime_config().dump_dir,
                            "layer_" + std::to_string(layer_idx) + "_mlp_mul",
                            device_id, act.mlp_gate,
                            seq_len, intermediate_size, compute_dtype);
        if (err) return err;
    } else {
        // Fast path: fuse SiLU + mul (SwiGLU) to cut one launch and one global write+read.
        if (compute_dtype == DType::BF16) {
            kernels::silu_mul_fused_bf16(
                static_cast<__nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        } else {
            kernels::silu_mul_fused_f16(
                static_cast<half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        }
    }
    
    // Down projection: [M, intermediate] @ [hidden, intermediate]^T = [M, hidden]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, M, intermediate_size,
        &alpha_one,
        layer.down_proj_weight, cuda_dtype, intermediate_size,
        act.mlp_gate, cuda_dtype, intermediate_size,
        &beta_zero,
        act.mlp_down, cuda_dtype, hidden_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_mlp_out",
                                     device_id, act.mlp_down,
                                     seq_len, hidden_size, compute_dtype);
        if (err) return err;
    }
    
    // =====================================================================
    // Residual Connection (MLP)
    // =====================================================================
    if (compute_dtype == DType::BF16) {
        kernels::elementwise_add_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(act.mlp_down),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            batch_size * seq_len * hidden_size,
            stream
        );
    } else {
        kernels::elementwise_add_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(act.mlp_down),
            static_cast<const half*>(act.hidden_states),
            batch_size * seq_len * hidden_size,
            stream
        );
    }

    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.ffn_end, stream));
    }
    
    // Check mode: dump last token hidden state after layer
    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            std::string name = "layer_" + std::to_string(layer_idx) + "_last_hidden";
            Error err = dump_last_row(session.runtime_config().dump_dir, name, device_id,
                                         act.hidden_states, seq_len, hidden_size, compute_dtype);
            if (err) return err;
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.end, stream));
        CUDA_CHECK(cudaEventSynchronize(ev.end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.start, ev.end));
        last_layer_profile_ms_[static_cast<size_t>(layer_idx)] = ms;
    }
    
    return Error::success();
}

Error CudaRuntime::forward_final_norm(int batch_size, int seq_len, Session& session) {
    int lm_device = device_map_.lm_head_device;
    auto& act = activations_[lm_device];
    auto stream = streams_[lm_device];
    DType compute_dtype = weights_.dtype;
    
    CUDA_CHECK(cudaSetDevice(lm_device));
    
    // 如果最后一层不在 lm_head 设备上，需要拷贝
    int last_layer_device = device_map_.layer_to_device[config_.num_layers - 1];
    if (last_layer_device != lm_device) {
        size_t size = batch_size * seq_len * config_.hidden_size * dtype_size(compute_dtype);
        auto t0 = std::chrono::high_resolution_clock::now();
        Error err = copy_bytes_peer_or_staged(act.hidden_states, lm_device,
                                              activations_[last_layer_device].hidden_states, last_layer_device,
                                              size);
        if (err) return err;
        if (profile_stages_) {
            auto t1 = std::chrono::high_resolution_clock::now();
            last_stage_profile_ms_.p2p_ms += static_cast<float>(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    
    if (profile_stages_) {
        ensure_simple_stage_events_(final_norm_events_, lm_device);
        CUDA_CHECK(cudaEventRecord(final_norm_events_.start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.final_norm),
            batch_size, seq_len, config_.hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(weights_.final_norm),
            batch_size, seq_len, config_.hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(final_norm_events_.end, stream));
    }

    if (session.runtime_config().check_correctness) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "final_norm_last_hidden",
                                     lm_device, act.norm_out,
                                     seq_len, config_.hidden_size, compute_dtype);
        if (err) return err;
    }
    
    return Error::success();
}

Error CudaRuntime::forward_lm_head(int batch_size, int seq_len) {
    int lm_device = device_map_.lm_head_device;
    auto& act = activations_[lm_device];
    auto stream = streams_[lm_device];
    auto& cublas = cublas_handles_[lm_device];
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    
    CUDA_CHECK(cudaSetDevice(lm_device));
    cublasSetStream(cublas.get(), stream);

    if (profile_stages_) {
        ensure_simple_stage_events_(lm_head_events_, lm_device);
        CUDA_CHECK(cudaEventRecord(lm_head_events_.start, stream));
    }
    
    // 取每个 batch 的最后一个 token hidden state，并打包为 [hidden_size, batch]（列主序列=batch）
    const int hidden_size = static_cast<int>(config_.hidden_size);
    const int vocab_size = static_cast<int>(config_.vocab_size);

    if (compute_dtype == DType::BF16) {
        kernels::copy_last_hidden_bf16(
            static_cast<__nv_bfloat16*>(act.last_hidden),
            static_cast<const __nv_bfloat16*>(act.norm_out),
            batch_size, seq_len, hidden_size,
            stream
        );
    } else {
        kernels::copy_last_hidden_f16(
            static_cast<half*>(act.last_hidden),
            static_cast<const half*>(act.norm_out),
            batch_size, seq_len, hidden_size,
            stream
        );
    }

    const void* hidden_ptr = static_cast<const void*>(act.last_hidden);
    const void* lm_head_ptr = static_cast<const void*>(weights_.lm_head);
    float* logits_ptr = static_cast<float*>(act.logits);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // GEMM: C = alpha * A * B + beta * C
    // A: [vocab_size, hidden_size], B: [hidden_size, batch], C: [vocab_size, batch]
    // 由于 cuBLAS 是列主序，需要调整参数
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,      // A^T * B
        vocab_size, batch_size, hidden_size,
        &alpha,
        lm_head_ptr, cuda_dtype, hidden_size,
        hidden_ptr, cuda_dtype, hidden_size,
        &beta,
        logits_ptr, CUDA_R_32F, vocab_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(lm_head_events_.end, stream));
    }
    
    return Error::success();
}

Error CudaRuntime::decode_batch(const std::vector<int>& last_tokens,
                                Session& session,
                                std::vector<float>& logits_flat) {
    EMBER_RETURN_IF_ERROR(decode_batch_to_device(last_tokens, session));

    const int batch_size = session.runtime_config().batch_size;
    const int lm_device = device_map_.lm_head_device;
    cuda_sync(lm_device);

    logits_flat.resize(static_cast<size_t>(batch_size) * static_cast<size_t>(config_.vocab_size));
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits_flat.data(), activations_[lm_device].logits,
                          logits_flat.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());
    return Error::success();
}

Error CudaRuntime::decode_batch_greedy(const std::vector<int>& last_tokens,
                                       Session& session,
                                       std::vector<int>& next_tokens) {
    EMBER_RETURN_IF_ERROR(decode_batch_to_device(last_tokens, session));

    const int batch_size = session.runtime_config().batch_size;
    const int vocab_size = static_cast<int>(config_.vocab_size);
    const int lm_device = device_map_.lm_head_device;

    CUDA_CHECK(cudaSetDevice(lm_device));
    if (!next_tokens_dev_ || next_tokens_cap_ < batch_size) {
        if (next_tokens_dev_) cuda_free(next_tokens_dev_);
        CUDA_CHECK(cudaMalloc(&next_tokens_dev_, static_cast<size_t>(batch_size) * sizeof(int)));
        next_tokens_cap_ = batch_size;
    }

    kernels::argmax_f32(
        next_tokens_dev_,
        static_cast<const float*>(activations_[lm_device].logits),
        batch_size,
        vocab_size,
        streams_[lm_device]
    );

    next_tokens.resize(static_cast<size_t>(batch_size));
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(next_tokens.data(), next_tokens_dev_,
                               static_cast<size_t>(batch_size) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               streams_[lm_device]));
    CUDA_CHECK(cudaStreamSynchronize(streams_[lm_device]));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());
    return Error::success();
}

}  // namespace cuda

// =============================================================================
// RuntimeFactory 实现（在 ember 命名空间）
// =============================================================================

std::unique_ptr<IRuntime> RuntimeFactory::create_cuda() {
    return std::make_unique<cuda::CudaRuntime>();
}

std::unique_ptr<IRuntime> RuntimeFactory::create_cpu() {
    // TODO: 实现 CPU 后端
    return nullptr;
}

std::unique_ptr<IRuntime> RuntimeFactory::create_auto() {
    auto cuda_runtime = create_cuda();
    if (cuda_runtime && cuda_runtime->available()) {
        return cuda_runtime;
    }
    return create_cpu();
}

}  // namespace ember
