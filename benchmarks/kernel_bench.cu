#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/kernels/kernels.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

bool cuda_ok(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    std::cerr << "cuda error (" << what << "): " << cudaGetErrorString(err) << "\n";
    return false;
}

struct BenchConfig {
    int device = 0;
    int iters = 100;
    int warmup = 10;
    double hw_bandwidth_gbps = 912.0;  // RTX 3080 Ti GDDR6X peak (approx)
    std::string dtype = "f16";         // f16|bf16
    bool include_embedding = false;
    std::string csv_path;
};

double median_us(std::vector<float>& ms) {
    if (ms.empty()) return 0.0;
    std::sort(ms.begin(), ms.end());
    const float med_ms = ms[ms.size() / 2];
    return static_cast<double>(med_ms) * 1000.0;
}

template <typename LaunchFn>
double bench_kernel_us(const BenchConfig& cfg,
                       cudaStream_t stream,
                       cudaEvent_t ev_start,
                       cudaEvent_t ev_end,
                       LaunchFn&& launch) {
    for (int i = 0; i < cfg.warmup; ++i) {
        launch(stream);
    }
    if (!cuda_ok(cudaStreamSynchronize(stream), "warmup sync")) die("warmup sync failed");

    std::vector<float> times_ms;
    times_ms.reserve(static_cast<size_t>(cfg.iters));
    for (int i = 0; i < cfg.iters; ++i) {
        if (!cuda_ok(cudaEventRecord(ev_start, stream), "event record start")) die("event record start failed");
        launch(stream);
        if (!cuda_ok(cudaEventRecord(ev_end, stream), "event record end")) die("event record end failed");
        if (!cuda_ok(cudaEventSynchronize(ev_end), "event sync")) die("event sync failed");
        float ms = 0.0f;
        if (!cuda_ok(cudaEventElapsedTime(&ms, ev_start, ev_end), "event elapsed")) die("event elapsed failed");
        times_ms.push_back(ms);
    }
    return median_us(times_ms);
}

struct Row {
    std::string kernel;
    std::string dtype;
    std::string shape;
    double elapsed_us = 0.0;
    double bytes_moved = 0.0;
    double effective_gbps = 0.0;
    double efficiency_pct = 0.0;
};

Row make_row(const BenchConfig& cfg,
             const std::string& kernel,
             const std::string& shape,
             double elapsed_us,
             double bytes_moved) {
    Row r;
    r.kernel = kernel;
    r.dtype = cfg.dtype;
    r.shape = shape;
    r.elapsed_us = elapsed_us;
    r.bytes_moved = bytes_moved;
    if (elapsed_us > 0.0) {
        const double seconds = elapsed_us * 1e-6;
        r.effective_gbps = (bytes_moved / seconds) / 1e9;
        r.efficiency_pct = (cfg.hw_bandwidth_gbps > 0.0) ? (r.effective_gbps / cfg.hw_bandwidth_gbps) * 100.0 : 0.0;
    }
    return r;
}

std::string fmt_shape(std::initializer_list<std::pair<const char*, int64_t>> kv) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& it : kv) {
        if (!first) oss << " ";
        first = false;
        oss << it.first << "=" << it.second;
    }
    return oss.str();
}

size_t elem_size_bytes(const BenchConfig& cfg) {
    if (cfg.dtype == "f16") return sizeof(half);
    if (cfg.dtype == "bf16") return sizeof(__nv_bfloat16);
    die("unsupported --dtype: " + cfg.dtype);
}

}  // namespace

int main(int argc, char** argv) {
    BenchConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Kernel Microbench (CUDA)\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " [options]\n\n"
                << "Options:\n"
                << "  --device N            CUDA device (default: 0)\n"
                << "  --dtype f16|bf16      compute dtype (default: f16)\n"
                << "  --iters N             timed iterations (default: 100)\n"
                << "  --warmup N            warmup iterations (default: 10)\n"
                << "  --hw-bw GBPS          roofline bandwidth (default: 912)\n"
                << "  --include-embedding   also benchmark embedding lookup (large alloc)\n"
                << "  --csv PATH            write CSV (default: stdout)\n";
            return 0;
        } else if (arg == "--device") {
            cfg.device = std::stoi(need("--device"));
        } else if (arg == "--dtype") {
            cfg.dtype = need("--dtype");
        } else if (arg == "--iters") {
            cfg.iters = std::stoi(need("--iters"));
        } else if (arg == "--warmup") {
            cfg.warmup = std::stoi(need("--warmup"));
        } else if (arg == "--hw-bw") {
            cfg.hw_bandwidth_gbps = std::stod(need("--hw-bw"));
        } else if (arg == "--include-embedding") {
            cfg.include_embedding = true;
        } else if (arg == "--csv") {
            cfg.csv_path = need("--csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (cfg.iters <= 0) die("--iters must be > 0");
    if (cfg.warmup < 0) die("--warmup must be >= 0");
    if (cfg.dtype != "f16" && cfg.dtype != "bf16") die("--dtype must be f16 or bf16");

    if (!cuda_ok(cudaSetDevice(cfg.device), "cudaSetDevice")) die("cudaSetDevice failed");

    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "cudaStreamCreate")) die("cudaStreamCreate failed");
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_end = nullptr;
    if (!cuda_ok(cudaEventCreate(&ev_start), "cudaEventCreate start")) die("cudaEventCreate failed");
    if (!cuda_ok(cudaEventCreate(&ev_end), "cudaEventCreate end")) die("cudaEventCreate failed");

    std::ostream* out = &std::cout;
    std::ofstream f;
    if (!cfg.csv_path.empty()) {
        f.open(cfg.csv_path);
        if (!f.is_open()) die("failed to open output: " + cfg.csv_path);
        out = &f;
    }

    *out << "kernel,dtype,shape,elapsed_us,bytes_moved,effective_gbps,efficiency_pct\n";

    auto emit = [&](const Row& r) {
        *out << r.kernel << "," << r.dtype << ",\"" << r.shape << "\","
             << std::fixed << std::setprecision(3) << r.elapsed_us << ","
             << std::fixed << std::setprecision(0) << r.bytes_moved << ","
             << std::fixed << std::setprecision(3) << r.effective_gbps << ","
             << std::fixed << std::setprecision(2) << r.efficiency_pct << "\n";
    };

    const size_t elem_sz = elem_size_bytes(cfg);

    // ---------------------------------------------------------------------
    // RMSNorm
    // ---------------------------------------------------------------------
    {
        const std::vector<int> hidden_sizes = {1024, 2048, 4096, 5120};
        const std::vector<int> seqs = {1, 128, 512, 2048};
        const int batch = 1;
        const float eps = 1e-6f;

        for (int hidden : hidden_sizes) {
            void* d_weight = nullptr;
            if (!cuda_ok(cudaMalloc(&d_weight, static_cast<size_t>(hidden) * elem_sz), "cudaMalloc weight")) die("oom");
            cuda_ok(cudaMemset(d_weight, 0, static_cast<size_t>(hidden) * elem_sz), "memset weight");

            for (int seq : seqs) {
                const int rows = batch * seq;
                const int64_t elems = static_cast<int64_t>(rows) * static_cast<int64_t>(hidden);
                void* d_in = nullptr;
                void* d_out = nullptr;
                if (!cuda_ok(cudaMalloc(&d_in, static_cast<size_t>(elems) * elem_sz), "cudaMalloc in")) die("oom");
                if (!cuda_ok(cudaMalloc(&d_out, static_cast<size_t>(elems) * elem_sz), "cudaMalloc out")) die("oom");
                cuda_ok(cudaMemset(d_in, 0, static_cast<size_t>(elems) * elem_sz), "memset in");

                const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                    if (cfg.dtype == "bf16") {
                        ember::cuda::kernels::rms_norm_bf16(
                            static_cast<__nv_bfloat16*>(d_out),
                            static_cast<const __nv_bfloat16*>(d_in),
                            static_cast<const __nv_bfloat16*>(d_weight),
                            batch,
                            seq,
                            hidden,
                            eps,
                            s);
                    } else {
                        ember::cuda::kernels::rms_norm_f16(
                            static_cast<half*>(d_out),
                            static_cast<const half*>(d_in),
                            static_cast<const half*>(d_weight),
                            batch,
                            seq,
                            hidden,
                            eps,
                            s);
                    }
                });

                const double bytes = static_cast<double>(elems) * static_cast<double>(elem_sz) * 3.0;
                emit(make_row(cfg, "rmsnorm", fmt_shape({{"seq", seq}, {"hidden", hidden}}), us, bytes));

                cudaFree(d_in);
                cudaFree(d_out);
            }
            cudaFree(d_weight);
        }
    }

    // ---------------------------------------------------------------------
    // Elementwise ops + SwiGLU fused helper
    // ---------------------------------------------------------------------
    {
        const int seq = 2048;
        const std::vector<int> hidden_sizes = {1024, 2048, 4096, 5120};
        const std::vector<int> intermediate_sizes = {3072, 6144, 12288, 17408};

        for (int hidden : hidden_sizes) {
            const int64_t elems = static_cast<int64_t>(seq) * static_cast<int64_t>(hidden);
            void* d_a = nullptr;
            void* d_b = nullptr;
            void* d_out = nullptr;
            if (!cuda_ok(cudaMalloc(&d_a, static_cast<size_t>(elems) * elem_sz), "cudaMalloc a")) die("oom");
            if (!cuda_ok(cudaMalloc(&d_b, static_cast<size_t>(elems) * elem_sz), "cudaMalloc b")) die("oom");
            if (!cuda_ok(cudaMalloc(&d_out, static_cast<size_t>(elems) * elem_sz), "cudaMalloc out")) die("oom");
            cuda_ok(cudaMemset(d_a, 0, static_cast<size_t>(elems) * elem_sz), "memset a");
            cuda_ok(cudaMemset(d_b, 0, static_cast<size_t>(elems) * elem_sz), "memset b");

            {
                const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                    if (cfg.dtype == "bf16") {
                        ember::cuda::kernels::silu_bf16(
                            static_cast<__nv_bfloat16*>(d_out),
                            static_cast<const __nv_bfloat16*>(d_a),
                            elems,
                            s);
                    } else {
                        ember::cuda::kernels::silu_f16(
                            static_cast<half*>(d_out),
                            static_cast<const half*>(d_a),
                            elems,
                            s);
                    }
                });
                const double bytes = static_cast<double>(elems) * static_cast<double>(elem_sz) * 2.0;
                emit(make_row(cfg, "silu", fmt_shape({{"seq", seq}, {"hidden", hidden}}), us, bytes));
            }

            {
                const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                    if (cfg.dtype == "bf16") {
                        ember::cuda::kernels::elementwise_add_bf16(
                            static_cast<__nv_bfloat16*>(d_out),
                            static_cast<const __nv_bfloat16*>(d_a),
                            static_cast<const __nv_bfloat16*>(d_b),
                            elems,
                            s);
                    } else {
                        ember::cuda::kernels::elementwise_add_f16(
                            static_cast<half*>(d_out),
                            static_cast<const half*>(d_a),
                            static_cast<const half*>(d_b),
                            elems,
                            s);
                    }
                });
                const double bytes = static_cast<double>(elems) * static_cast<double>(elem_sz) * 3.0;
                emit(make_row(cfg, "add", fmt_shape({{"seq", seq}, {"hidden", hidden}}), us, bytes));
            }

            {
                const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                    if (cfg.dtype == "bf16") {
                        ember::cuda::kernels::elementwise_mul_bf16(
                            static_cast<__nv_bfloat16*>(d_out),
                            static_cast<const __nv_bfloat16*>(d_a),
                            static_cast<const __nv_bfloat16*>(d_b),
                            elems,
                            s);
                    } else {
                        ember::cuda::kernels::elementwise_mul_f16(
                            static_cast<half*>(d_out),
                            static_cast<const half*>(d_a),
                            static_cast<const half*>(d_b),
                            elems,
                            s);
                    }
                });
                const double bytes = static_cast<double>(elems) * static_cast<double>(elem_sz) * 3.0;
                emit(make_row(cfg, "mul", fmt_shape({{"seq", seq}, {"hidden", hidden}}), us, bytes));
            }

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_out);
        }

        for (int inter : intermediate_sizes) {
            const int64_t elems = static_cast<int64_t>(seq) * static_cast<int64_t>(inter);
            void* d_gate = nullptr;
            void* d_up = nullptr;
            if (!cuda_ok(cudaMalloc(&d_gate, static_cast<size_t>(elems) * elem_sz), "cudaMalloc gate")) die("oom");
            if (!cuda_ok(cudaMalloc(&d_up, static_cast<size_t>(elems) * elem_sz), "cudaMalloc up")) die("oom");
            cuda_ok(cudaMemset(d_gate, 0, static_cast<size_t>(elems) * elem_sz), "memset gate");
            cuda_ok(cudaMemset(d_up, 0, static_cast<size_t>(elems) * elem_sz), "memset up");

            const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                if (cfg.dtype == "bf16") {
                    ember::cuda::kernels::silu_mul_fused_bf16(
                        static_cast<__nv_bfloat16*>(d_gate),
                        static_cast<const __nv_bfloat16*>(d_up),
                        elems,
                        s);
                } else {
                    ember::cuda::kernels::silu_mul_fused_f16(
                        static_cast<half*>(d_gate),
                        static_cast<const half*>(d_up),
                        elems,
                        s);
                }
            });
            const double bytes = static_cast<double>(elems) * static_cast<double>(elem_sz) * 3.0;
            emit(make_row(cfg, "silu_mul_fused", fmt_shape({{"seq", seq}, {"inter", inter}}), us, bytes));

            cudaFree(d_gate);
            cudaFree(d_up);
        }
    }

    // ---------------------------------------------------------------------
    // RoPE (in-place)
    // ---------------------------------------------------------------------
    {
        const int batch = 1;
        const int num_heads = 32;
        const int num_kv_heads = 8;
        const std::vector<int> head_dims = {64, 80, 128};
        const std::vector<int> seqs = {1, 128, 512};
        const int start_pos = 0;
        const float theta = 10000.0f;

        for (int head_dim : head_dims) {
            for (int seq : seqs) {
                const int64_t q_elems =
                    static_cast<int64_t>(batch) * static_cast<int64_t>(seq) * static_cast<int64_t>(num_heads) *
                    static_cast<int64_t>(head_dim);
                const int64_t k_elems =
                    static_cast<int64_t>(batch) * static_cast<int64_t>(seq) * static_cast<int64_t>(num_kv_heads) *
                    static_cast<int64_t>(head_dim);
                void* d_q = nullptr;
                void* d_k = nullptr;
                if (!cuda_ok(cudaMalloc(&d_q, static_cast<size_t>(q_elems) * elem_sz), "cudaMalloc q")) die("oom");
                if (!cuda_ok(cudaMalloc(&d_k, static_cast<size_t>(k_elems) * elem_sz), "cudaMalloc k")) die("oom");
                cuda_ok(cudaMemset(d_q, 0, static_cast<size_t>(q_elems) * elem_sz), "memset q");
                cuda_ok(cudaMemset(d_k, 0, static_cast<size_t>(k_elems) * elem_sz), "memset k");

                const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                    if (cfg.dtype == "bf16") {
                        ember::cuda::kernels::apply_rope_bf16(
                            static_cast<__nv_bfloat16*>(d_q),
                            static_cast<__nv_bfloat16*>(d_k),
                            batch,
                            seq,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            start_pos,
                            theta,
                            s);
                    } else {
                        ember::cuda::kernels::apply_rope_f16(
                            static_cast<half*>(d_q),
                            static_cast<half*>(d_k),
                            batch,
                            seq,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            start_pos,
                            theta,
                            s);
                    }
                });
                const double bytes = static_cast<double>(q_elems + k_elems) * static_cast<double>(elem_sz) * 2.0;
                emit(make_row(cfg,
                              "rope",
                              fmt_shape({{"seq", seq}, {"heads", num_heads}, {"kv_heads", num_kv_heads}, {"d", head_dim}}),
                              us,
                              bytes));

                cudaFree(d_q);
                cudaFree(d_k);
            }
        }
    }

    // ---------------------------------------------------------------------
    // KV cache update
    // ---------------------------------------------------------------------
    {
        const int batch = 1;
        const int num_kv_heads = 8;
        const int head_dim = 128;
        const int seq_len = 128;
        const int max_seq = 2048;
        const int start_pos = 256;

        const int64_t new_elems =
            static_cast<int64_t>(batch) * static_cast<int64_t>(seq_len) * static_cast<int64_t>(num_kv_heads) *
            static_cast<int64_t>(head_dim);
        const int64_t cache_elems =
            static_cast<int64_t>(batch) * static_cast<int64_t>(num_kv_heads) * static_cast<int64_t>(max_seq) *
            static_cast<int64_t>(head_dim);

        void* d_k_cache = nullptr;
        void* d_v_cache = nullptr;
        void* d_k_new = nullptr;
        void* d_v_new = nullptr;
        if (!cuda_ok(cudaMalloc(&d_k_cache, static_cast<size_t>(cache_elems) * elem_sz), "cudaMalloc k_cache")) die("oom");
        if (!cuda_ok(cudaMalloc(&d_v_cache, static_cast<size_t>(cache_elems) * elem_sz), "cudaMalloc v_cache")) die("oom");
        if (!cuda_ok(cudaMalloc(&d_k_new, static_cast<size_t>(new_elems) * elem_sz), "cudaMalloc k_new")) die("oom");
        if (!cuda_ok(cudaMalloc(&d_v_new, static_cast<size_t>(new_elems) * elem_sz), "cudaMalloc v_new")) die("oom");
        cuda_ok(cudaMemset(d_k_cache, 0, static_cast<size_t>(cache_elems) * elem_sz), "memset k_cache");
        cuda_ok(cudaMemset(d_v_cache, 0, static_cast<size_t>(cache_elems) * elem_sz), "memset v_cache");
        cuda_ok(cudaMemset(d_k_new, 0, static_cast<size_t>(new_elems) * elem_sz), "memset k_new");
        cuda_ok(cudaMemset(d_v_new, 0, static_cast<size_t>(new_elems) * elem_sz), "memset v_new");

        const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
            if (cfg.dtype == "bf16") {
                ember::cuda::kernels::update_kv_cache_bf16(
                    static_cast<__nv_bfloat16*>(d_k_cache),
                    static_cast<__nv_bfloat16*>(d_v_cache),
                    static_cast<const __nv_bfloat16*>(d_k_new),
                    static_cast<const __nv_bfloat16*>(d_v_new),
                    batch,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    start_pos,
                    max_seq,
                    s);
            } else {
                ember::cuda::kernels::update_kv_cache_f16(
                    static_cast<half*>(d_k_cache),
                    static_cast<half*>(d_v_cache),
                    static_cast<const half*>(d_k_new),
                    static_cast<const half*>(d_v_new),
                    batch,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    start_pos,
                    max_seq,
                    s);
            }
        });
        const double bytes = static_cast<double>(new_elems) * static_cast<double>(elem_sz) * 4.0;
        emit(make_row(cfg,
                      "kv_update",
                      fmt_shape({{"seq", seq_len}, {"kv_heads", num_kv_heads}, {"d", head_dim}, {"max_seq", max_seq}}),
                      us,
                      bytes));

        cudaFree(d_k_cache);
        cudaFree(d_v_cache);
        cudaFree(d_k_new);
        cudaFree(d_v_new);
    }

    // ---------------------------------------------------------------------
    // Embedding lookup (optional; large allocation)
    // ---------------------------------------------------------------------
    if (cfg.include_embedding) {
        const int vocab_size = 152064;
        const std::vector<int> hidden_sizes = {1024, 2048, 4096, 5120};
        const int batch = 1;
        const int seq = 512;

        for (int hidden : hidden_sizes) {
            const int64_t emb_elems = static_cast<int64_t>(vocab_size) * static_cast<int64_t>(hidden);
            const int64_t out_elems = static_cast<int64_t>(batch) * static_cast<int64_t>(seq) * static_cast<int64_t>(hidden);

            void* d_emb = nullptr;
            void* d_out = nullptr;
            int* d_ids = nullptr;
            if (!cuda_ok(cudaMalloc(&d_emb, static_cast<size_t>(emb_elems) * elem_sz), "cudaMalloc emb")) die("oom");
            if (!cuda_ok(cudaMalloc(&d_out, static_cast<size_t>(out_elems) * elem_sz), "cudaMalloc out")) die("oom");
            if (!cuda_ok(cudaMalloc(&d_ids, static_cast<size_t>(batch) * static_cast<size_t>(seq) * sizeof(int)),
                        "cudaMalloc ids"))
                die("oom");

            cuda_ok(cudaMemset(d_emb, 0, static_cast<size_t>(emb_elems) * elem_sz), "memset emb");
            cuda_ok(cudaMemset(d_out, 0, static_cast<size_t>(out_elems) * elem_sz), "memset out");

            std::vector<int> ids(static_cast<size_t>(batch) * static_cast<size_t>(seq));
            for (size_t i = 0; i < ids.size(); ++i) {
                ids[i] = static_cast<int>(i % static_cast<size_t>(vocab_size));
            }
            if (!cuda_ok(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice), "memcpy ids")) {
                die("memcpy ids failed");
            }

            const double us = bench_kernel_us(cfg, stream, ev_start, ev_end, [&](cudaStream_t s) {
                if (cfg.dtype == "bf16") {
                    ember::cuda::kernels::embedding_lookup_bf16(
                        static_cast<__nv_bfloat16*>(d_out),
                        static_cast<const __nv_bfloat16*>(d_emb),
                        d_ids,
                        batch,
                        seq,
                        hidden,
                        s);
                } else {
                    ember::cuda::kernels::embedding_lookup_f16(
                        static_cast<half*>(d_out),
                        static_cast<const half*>(d_emb),
                        d_ids,
                        batch,
                        seq,
                        hidden,
                        s);
                }
            });

            const double bytes =
                static_cast<double>(out_elems) * static_cast<double>(elem_sz) * 2.0 +
                static_cast<double>(batch) * static_cast<double>(seq) * sizeof(int);
            emit(make_row(cfg, "embedding", fmt_shape({{"seq", seq}, {"hidden", hidden}, {"vocab", vocab_size}}), us, bytes));

            cudaFree(d_emb);
            cudaFree(d_out);
            cudaFree(d_ids);
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);
    return 0;
}

