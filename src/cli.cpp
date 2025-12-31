#include "cli.h"
#include "utils.h"
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <algorithm>

namespace ember {

SplitMode parse_split_mode(const std::string& s) {
    if (s == "none") return SplitMode::NONE;
    if (s == "layer") return SplitMode::LAYER;
    if (s == "row") return SplitMode::ROW;
    LOG_ERROR("未知的 split-mode: %s，使用默认值 layer", s.c_str());
    return SplitMode::LAYER;
}

const char* split_mode_to_string(SplitMode mode) {
    switch (mode) {
        case SplitMode::NONE:  return "none";
        case SplitMode::LAYER: return "layer";
        case SplitMode::ROW:   return "row";
    }
    return "unknown";
}

bool CliArgs::validate() const {
    if (model_path.empty()) {
        LOG_ERROR("错误: 必须指定模型路径 --model");
        return false;
    }
    
    if (main_gpu < 0) {
        LOG_ERROR("错误: --main-gpu 必须 >= 0");
        return false;
    }
    
    if (n_ctx < 128 || n_ctx > 131072) {
        LOG_ERROR("错误: --ctx-size 必须在 128-131072 之间");
        return false;
    }
    
    if (n_batch < 1 || n_batch > n_ctx) {
        LOG_ERROR("错误: --batch-size 必须在 1-%d 之间", n_ctx);
        return false;
    }
    
    if (temperature < 0.0f) {
        LOG_ERROR("错误: --temp 必须 >= 0");
        return false;
    }
    
    // tensor_split 校验在模型加载时进行（需要知道 GPU 数量）
    
    return true;
}

void CliArgs::print() const {
    LOG_INFO("========== Ember 配置 ==========");
    LOG_INFO("模型路径: %s", model_path.c_str());
    LOG_INFO("GPU 层数: %d (-1 表示全部)", n_gpu_layers);
    LOG_INFO("主 GPU: %d", main_gpu);
    LOG_INFO("分配模式: %s", split_mode_to_string(split_mode));
    
    if (!tensor_split.empty()) {
        std::ostringstream oss;
        for (size_t i = 0; i < tensor_split.size(); ++i) {
            if (i > 0) oss << ",";
            oss << tensor_split[i];
        }
        LOG_INFO("Tensor 分配: %s", oss.str().c_str());
    }
    
    LOG_INFO("上下文长度: %d", n_ctx);
    LOG_INFO("Batch 大小: %d", n_batch);
    LOG_INFO("CPU 线程: %d", n_threads);
    LOG_INFO("生成数量: %d", n_predict);
    LOG_INFO("Temperature: %.2f", temperature);
    LOG_INFO("Top-P: %.2f", top_p);
    LOG_INFO("Top-K: %d", top_k);
    LOG_INFO("====================================");
}

void print_help(const char* prog_name) {
    printf("Ember - Qwen3 CUDA-only 推理引擎\n\n");
    printf("用法: %s [选项] -m <模型路径>\n\n", prog_name);
    printf("必需参数:\n");
    printf("  -m, --model PATH        GGUF 模型文件路径\n\n");
    printf("GPU 参数:\n");
    printf("  -ngl, --n-gpu-layers N  上 GPU 的层数 (-1 = 全部, 默认: -1)\n");
    printf("  --main-gpu N            主 GPU 索引 (默认: 0)\n");
    printf("  --split-mode MODE       多 GPU 分配: none/layer/row (默认: layer)\n");
    printf("  --tensor-split F,F,...  GPU 显存分配比例 (如: 0.5,0.5)\n\n");
    printf("推理参数:\n");
    printf("  -c, --ctx-size N        上下文长度 (默认: 2048)\n");
    printf("  -b, --batch-size N      Batch 大小 (默认: 512)\n");
    printf("  -t, --threads N         CPU 线程数 (默认: 4)\n\n");
    printf("生成参数:\n");
    printf("  -n, --n-predict N       生成 token 数 (默认: 128)\n");
    printf("  --temp F                Temperature (默认: 0.7)\n");
    printf("  --top-p F               Top-P 采样 (默认: 0.9)\n");
    printf("  --top-k N               Top-K 采样 (默认: 40)\n\n");
    printf("其他:\n");
    printf("  -p, --prompt TEXT       输入提示词\n");
    printf("  -i, --interactive       交互模式\n");
    printf("  -v, --verbose           详细日志\n");
    printf("  -h, --help              显示帮助\n\n");
    printf("示例:\n");
    printf("  %s -m qwen3-4b.gguf -p \"你好\"\n", prog_name);
    printf("  %s -m qwen3-4b.gguf -ngl -1 --split-mode layer --tensor-split 0.6,0.4\n", prog_name);
}

// 解析逗号分隔的浮点数列表
static std::vector<float> parse_float_list(const char* s) {
    std::vector<float> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stof(token));
    }
    return result;
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        
        // 帮助
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            args.help = true;
            return args;
        }
        
        // 模型路径
        if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0) {
            if (++i >= argc) { LOG_ERROR("--model 需要参数"); exit(1); }
            args.model_path = argv[i];
            continue;
        }
        
        // GPU 层数
        if (strcmp(arg, "-ngl") == 0 || strcmp(arg, "--n-gpu-layers") == 0) {
            if (++i >= argc) { LOG_ERROR("--n-gpu-layers 需要参数"); exit(1); }
            args.n_gpu_layers = std::atoi(argv[i]);
            continue;
        }
        
        // 主 GPU
        if (strcmp(arg, "--main-gpu") == 0) {
            if (++i >= argc) { LOG_ERROR("--main-gpu 需要参数"); exit(1); }
            args.main_gpu = std::atoi(argv[i]);
            continue;
        }
        
        // 分配模式
        if (strcmp(arg, "--split-mode") == 0) {
            if (++i >= argc) { LOG_ERROR("--split-mode 需要参数"); exit(1); }
            args.split_mode = parse_split_mode(argv[i]);
            continue;
        }
        
        // Tensor 分配
        if (strcmp(arg, "--tensor-split") == 0) {
            if (++i >= argc) { LOG_ERROR("--tensor-split 需要参数"); exit(1); }
            args.tensor_split = parse_float_list(argv[i]);
            continue;
        }
        
        // 上下文长度
        if (strcmp(arg, "-c") == 0 || strcmp(arg, "--ctx-size") == 0) {
            if (++i >= argc) { LOG_ERROR("--ctx-size 需要参数"); exit(1); }
            args.n_ctx = std::atoi(argv[i]);
            continue;
        }
        
        // Batch 大小
        if (strcmp(arg, "-b") == 0 || strcmp(arg, "--batch-size") == 0) {
            if (++i >= argc) { LOG_ERROR("--batch-size 需要参数"); exit(1); }
            args.n_batch = std::atoi(argv[i]);
            continue;
        }
        
        // 线程数
        if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) {
            if (++i >= argc) { LOG_ERROR("--threads 需要参数"); exit(1); }
            args.n_threads = std::atoi(argv[i]);
            continue;
        }
        
        // 生成数量
        if (strcmp(arg, "-n") == 0 || strcmp(arg, "--n-predict") == 0) {
            if (++i >= argc) { LOG_ERROR("--n-predict 需要参数"); exit(1); }
            args.n_predict = std::atoi(argv[i]);
            continue;
        }
        
        // Temperature
        if (strcmp(arg, "--temp") == 0) {
            if (++i >= argc) { LOG_ERROR("--temp 需要参数"); exit(1); }
            args.temperature = std::stof(argv[i]);
            continue;
        }
        
        // Top-P
        if (strcmp(arg, "--top-p") == 0) {
            if (++i >= argc) { LOG_ERROR("--top-p 需要参数"); exit(1); }
            args.top_p = std::stof(argv[i]);
            continue;
        }
        
        // Top-K
        if (strcmp(arg, "--top-k") == 0) {
            if (++i >= argc) { LOG_ERROR("--top-k 需要参数"); exit(1); }
            args.top_k = std::atoi(argv[i]);
            continue;
        }
        
        // 提示词
        if (strcmp(arg, "-p") == 0 || strcmp(arg, "--prompt") == 0) {
            if (++i >= argc) { LOG_ERROR("--prompt 需要参数"); exit(1); }
            args.prompt = argv[i];
            continue;
        }
        
        // 交互模式
        if (strcmp(arg, "-i") == 0 || strcmp(arg, "--interactive") == 0) {
            args.interactive = true;
            continue;
        }
        
        // 详细日志
        if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            args.verbose = true;
            continue;
        }
        
        LOG_WARN("未知参数: %s", arg);
    }
    
    return args;
}

} // namespace ember
