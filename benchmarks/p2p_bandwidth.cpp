#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ',')) {
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

bool ends_with(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    return std::memcmp(s.data() + (s.size() - suf.size()), suf.data(), suf.size()) == 0;
}

size_t parse_size(const std::string& token) {
    if (token.empty()) die("empty size token");
    std::string s = token;
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    size_t mul = 1;
    if (ends_with(s, "kb")) {
        mul = 1024;
        s.resize(s.size() - 2);
    } else if (ends_with(s, "k")) {
        mul = 1024;
        s.resize(s.size() - 1);
    } else if (ends_with(s, "mb")) {
        mul = 1024 * 1024;
        s.resize(s.size() - 2);
    } else if (ends_with(s, "m")) {
        mul = 1024 * 1024;
        s.resize(s.size() - 1);
    } else if (ends_with(s, "gb")) {
        mul = 1024ull * 1024ull * 1024ull;
        s.resize(s.size() - 2);
    } else if (ends_with(s, "g")) {
        mul = 1024ull * 1024ull * 1024ull;
        s.resize(s.size() - 1);
    }

    if (s.empty()) die("invalid size token: " + token);
    char* end = nullptr;
    unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (!end || *end != '\0') die("invalid size token: " + token);
    return static_cast<size_t>(v) * mul;
}

std::vector<size_t> parse_sizes(const std::string& s) {
    std::vector<size_t> out;
    for (const auto& tok : split_csv(s)) out.push_back(parse_size(tok));
    if (out.empty()) die("no sizes specified");
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> parse_gpus(const std::string& s) {
    auto toks = split_csv(s);
    if (toks.size() != 2) die("--gpus expects exactly 2 ids, e.g. 0,1");
    return {std::stoi(toks[0]), std::stoi(toks[1])};
}

bool can_p2p(int src, int dst) {
    int can = 0;
    cudaDeviceCanAccessPeer(&can, dst, src);
    return can != 0;
}

void enable_p2p(int src, int dst) {
    cudaSetDevice(dst);
    cudaError_t err = cudaDeviceEnablePeerAccess(src, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return;
    }
    if (err != cudaSuccess) {
        die(std::string("cudaDeviceEnablePeerAccess failed: ") + cudaGetErrorString(err));
    }
}

struct TimedResult {
    double avg_us = 0.0;
    double gbps = 0.0;
};

TimedResult measure_memcpy_peer_async(int src, int dst, size_t bytes, int iters, int warmup) {
    cudaSetDevice(src);
    void* d_src = nullptr;
    if (cudaMalloc(&d_src, bytes) != cudaSuccess) die("cudaMalloc(src) failed");

    cudaSetDevice(dst);
    void* d_dst = nullptr;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) die("cudaMalloc(dst) failed");

    cudaStream_t stream;
    cudaSetDevice(dst);
    if (cudaStreamCreate(&stream) != cudaSuccess) die("cudaStreamCreate failed");

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    for (int i = 0; i < warmup; ++i) {
        cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes, stream);
    }
    cudaEventRecord(ev0, stream);
    for (int i = 0; i < iters; ++i) {
        cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes, stream);
    }
    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_src);
    cudaSetDevice(dst);
    cudaFree(d_dst);

    const double total_us = static_cast<double>(ms) * 1000.0;
    const double avg_us = total_us / static_cast<double>(iters);
    const double gbps = (static_cast<double>(bytes) / (avg_us * 1e-6)) / 1e9;
    return {avg_us, gbps};
}

TimedResult measure_staged_via_host(int src, int dst, size_t bytes, int iters, int warmup) {
    cudaSetDevice(src);
    void* d_src = nullptr;
    if (cudaMalloc(&d_src, bytes) != cudaSuccess) die("cudaMalloc(src) failed");

    cudaSetDevice(dst);
    void* d_dst = nullptr;
    if (cudaMalloc(&d_dst, bytes) != cudaSuccess) die("cudaMalloc(dst) failed");

    void* h_buf = nullptr;
    if (cudaMallocHost(&h_buf, bytes) != cudaSuccess) die("cudaMallocHost failed");

    cudaSetDevice(src);
    cudaStream_t s_src;
    cudaStreamCreate(&s_src);

    cudaSetDevice(dst);
    cudaStream_t s_dst;
    cudaStreamCreate(&s_dst);

    auto one_iter = [&]() {
        cudaSetDevice(src);
        cudaMemcpyAsync(h_buf, d_src, bytes, cudaMemcpyDeviceToHost, s_src);
        cudaStreamSynchronize(s_src);
        cudaSetDevice(dst);
        cudaMemcpyAsync(d_dst, h_buf, bytes, cudaMemcpyHostToDevice, s_dst);
        cudaStreamSynchronize(s_dst);
    };

    for (int i = 0; i < warmup; ++i) one_iter();

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) one_iter();
    const auto t1 = std::chrono::high_resolution_clock::now();

    const double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double avg_us = total_us / static_cast<double>(iters);
    const double gbps = (static_cast<double>(bytes) / (avg_us * 1e-6)) / 1e9;

    cudaSetDevice(src);
    cudaStreamDestroy(s_src);
    cudaSetDevice(dst);
    cudaStreamDestroy(s_dst);
    cudaFreeHost(h_buf);
    cudaSetDevice(src);
    cudaFree(d_src);
    cudaSetDevice(dst);
    cudaFree(d_dst);

    return {avg_us, gbps};
}

void print_usage(const char* prog) {
    std::cout
        << "Ember P2P Bandwidth Benchmark\n\n"
        << "Usage:\n"
        << "  " << prog << " --gpus 0,1 --sizes 1k,10k,100k,1m,10m,100m [options]\n\n"
        << "Options:\n"
        << "  --gpus A,B        GPU ids (default: 0,1)\n"
        << "  --sizes LIST      Comma-separated sizes (e.g. 1024,1k,1m)\n"
        << "  --iters N         Timed iterations per size (default: 200)\n"
        << "  --warmup N        Warmup iterations (default: 20)\n"
        << "  --method both|p2p|staged   Compare methods (default: both)\n"
        << "  --direction both|a2b|b2a   (default: both)\n"
        << "  --hidden-sizes LIST        Emit per-token activation sizes (FP16) (e.g. 2560,4096)\n"
        << "  --csv PATH        Write CSV to file (default: stdout)\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int> gpus = {0, 1};
    std::vector<size_t> sizes;
    int iters = 200;
    int warmup = 20;
    std::string method = "both";
    std::string direction = "both";
    std::string csv_path;
    std::vector<int> hidden_sizes;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--gpus") {
            gpus = parse_gpus(need("--gpus"));
        } else if (arg == "--sizes") {
            sizes = parse_sizes(need("--sizes"));
        } else if (arg == "--iters") {
            iters = std::stoi(need("--iters"));
        } else if (arg == "--warmup") {
            warmup = std::stoi(need("--warmup"));
        } else if (arg == "--method") {
            method = need("--method");
        } else if (arg == "--direction") {
            direction = need("--direction");
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else if (arg == "--hidden-sizes") {
            for (const auto& tok : split_csv(need("--hidden-sizes"))) {
                hidden_sizes.push_back(std::stoi(tok));
            }
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (sizes.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    int dev_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess) {
        die(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(derr));
    }
    if (dev_count <= 0) die("no CUDA devices found");
    for (int d : gpus) {
        if (d < 0 || d >= dev_count) die("invalid GPU id: " + std::to_string(d));
    }

    const int a = gpus[0];
    const int b = gpus[1];

    bool p2p_ab = can_p2p(a, b);
    bool p2p_ba = can_p2p(b, a);

    if ((method == "p2p" || method == "both") && (!p2p_ab || !p2p_ba)) {
        std::cerr << "warning: peer access not available both ways (a->b="
                  << (p2p_ab ? "yes" : "no") << ", b->a=" << (p2p_ba ? "yes" : "no")
                  << "); p2p results may fail\n";
    }

    if (method == "p2p" || method == "both") {
        if (p2p_ab) enable_p2p(a, b);
        if (p2p_ba) enable_p2p(b, a);
    }

    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!csv_path.empty()) {
        file.open(csv_path);
        if (!file.is_open()) die("failed to open csv file: " + csv_path);
        out = &file;
    }

    *out << "data_size_bytes,transfer_time_us,bandwidth_gbps,direction,method\n";

    auto run_dir = [&](int src, int dst, const std::string& dir_name) {
        for (size_t bytes : sizes) {
            if (method == "p2p" || method == "both") {
                TimedResult r = measure_memcpy_peer_async(src, dst, bytes, iters, warmup);
                *out << bytes << "," << std::fixed << std::setprecision(3) << r.avg_us << ","
                     << std::setprecision(3) << r.gbps << "," << dir_name << ",cudaMemcpyPeerAsync\n";
            }
            if (method == "staged" || method == "both") {
                TimedResult r = measure_staged_via_host(src, dst, bytes, iters, warmup);
                *out << bytes << "," << std::fixed << std::setprecision(3) << r.avg_us << ","
                     << std::setprecision(3) << r.gbps << "," << dir_name << ",staged_d2h_h2d\n";
            }
        }
    };

    if (direction == "both" || direction == "a2b") {
        run_dir(a, b, "gpu" + std::to_string(a) + "_to_gpu" + std::to_string(b));
    }
    if (direction == "both" || direction == "b2a") {
        run_dir(b, a, "gpu" + std::to_string(b) + "_to_gpu" + std::to_string(a));
    }

    if (!hidden_sizes.empty()) {
        *out << "\n# activation_size_per_token_fp16\n";
        *out << "# hidden_size,bytes_per_token\n";
        for (int h : hidden_sizes) {
            const size_t bytes = static_cast<size_t>(h) * 2;  // FP16
            *out << "# " << h << "," << bytes << "\n";
        }
    }

    return 0;
}
