#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace ember {

// 多 GPU 分配模式
enum class SplitMode {
    NONE,   // 单 GPU
    LAYER,  // 按层分配
    ROW     // 按行分配（中间结果在 main_gpu）
};

// CLI 参数结构
struct CliArgs {
    // 模型参数
    std::string model_path;
    
    // GPU 参数
    int32_t n_gpu_layers = -1;          // -1 = 全部层上 GPU
    int32_t main_gpu = 0;               // 主 GPU 索引
    SplitMode split_mode = SplitMode::LAYER;
    std::vector<float> tensor_split;    // GPU 显存分配比例
    
    // 推理参数
    int32_t n_ctx = 2048;               // 上下文长度
    int32_t n_batch = 512;              // batch 大小
    int32_t n_threads = 4;              // CPU 线程数
    
    // 生成参数
    int32_t n_predict = 128;            // 生成 token 数
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    
    // 输入
    std::string prompt;
    
    // 运行模式
    bool interactive = false;           // 交互模式
    bool verbose = false;               // 详细日志
    bool help = false;
    
    // 验证参数有效性
    bool validate() const;
    
    // 打印配置
    void print() const;
};

// 解析命令行参数
CliArgs parse_args(int argc, char** argv);

// 打印帮助信息
void print_help(const char* prog_name);

// split_mode 字符串转换
SplitMode parse_split_mode(const std::string& s);
const char* split_mode_to_string(SplitMode mode);

} // namespace ember
