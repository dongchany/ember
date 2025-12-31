#include "cli.h"
#include "model.h"
#include "inference.h"
#include "utils.h"
#include <cuda_runtime.h>

using namespace ember;

int main(int argc, char** argv) {
    // 解析命令行参数
    CliArgs args = parse_args(argc, argv);
    
    if (args.help) {
        print_help(argv[0]);
        return 0;
    }
    
    // 设置日志级别
    if (args.verbose) {
        set_log_level(LogLevel::DEBUG);
    }
    
    // 验证参数
    if (!args.validate()) {
        return 1;
    }
    
    // 打印配置
    args.print();
    
    // 检查 CUDA 环境
    int gpu_count = get_gpu_count();
    if (gpu_count == 0) {
        LOG_ERROR("未检测到 CUDA GPU，Ember 仅支持 CUDA");
        LOG_ERROR("请确保已安装 NVIDIA 驱动和 CUDA");
        return 1;
    }
    
    // 打印 GPU 信息
    print_gpu_info();
    
    // 加载模型
    Qwen3Model& model = get_model();
    Error err = model.load(args.model_path, args);
    if (err != Error::OK) {
        LOG_ERROR("模型加载失败: %s", error_to_string(err));
        return 1;
    }
    
    // 打印模型信息
    model.print_info();
    
    // 初始化推理上下文
    InferenceContext ctx;
    err = ctx.init(&model, args);
    if (err != Error::OK) {
        LOG_ERROR("推理上下文初始化失败: %s", error_to_string(err));
        return 1;
    }
    
    // 运行推理
    if (args.interactive) {
        // 交互模式
        err = interactive_loop(ctx);
    } else if (!args.prompt.empty()) {
        // 单次生成
        GenerateResult result;
        err = generate(ctx, args.prompt, args.n_predict, result);
        
        if (err == Error::OK) {
            LOG_INFO("\n===== 生成统计 =====");
            LOG_INFO("总 token: %d", result.n_tokens);
            LOG_INFO("耗时: %.2f ms", result.time_ms);
            LOG_INFO("速度: %.2f tokens/s", result.tokens_per_second);
        }
    } else {
        LOG_ERROR("请指定 --prompt 或使用 --interactive 模式");
        print_help(argv[0]);
        return 1;
    }
    
    if (err != Error::OK) {
        LOG_ERROR("推理失败: %s", error_to_string(err));
        return 1;
    }
    
    LOG_INFO("完成");
    return 0;
}
