#include <iostream>
#include <vector>
#include <sstream>
#include "backends/cuda/cuda_utils.h"


// ANSI color helpers for the startup banner.
#define C_RESET "\033[0m"
#define C_ORANGE "\033[38;5;208m"
#define C_YELLOW "\033[33m"
#define C_RED "\033[31m"
#define C_DIM "\033[2m"
#define C_BOLD "\033[1m"

static inline void ember_banner() {
    std::printf("\n");
    std::printf(C_ORANGE  "    ███████╗███╗   ███╗██████╗ ███████╗██████╗ \n" C_RESET);
    std::printf(C_ORANGE  "    ██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗\n" C_RESET);
    std::printf(C_YELLOW  "    █████╗  ██╔████╔██║██████╔╝█████╗  ██████╔╝\n" C_RESET);
    std::printf(C_YELLOW  "    ██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██╔══██╗\n" C_RESET);
    std::printf(C_RED     "    ███████╗██║ ╚═╝ ██║██████╔╝███████╗██║  ██║\n" C_RESET);
    std::printf(C_RED     "    ╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝\n" C_RESET);
    std::printf("\n");
    std::printf(C_DIM     "    ─────────────────────────────────────────────\n" C_RESET);
    std::printf(C_BOLD    "      日拱一卒，功不唐捐；蹄疾步稳，如临深渊。\n" C_RESET);
    std::printf(C_DIM     "    ─────────────────────────────────────────────\n" C_RESET);
    std::printf(C_DIM     "    Lightweight CUDA Inference Engine for Qwen3\n" C_RESET);
    std::printf("\n");
}


struct Args {
    std::string model_path;
    std::string prompt;
    std::vector<int> devices = {0};
    int ctx_size = 2048;
    int n_predict = 128;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int no_repeat_ngram = 0;
    float memory_fraction = 0.9f;
    bool verbose = false;
    bool interactive = false;
    bool check_mode = false;
    bool temperature_set = false;
    bool top_p_set = false;
    bool top_k_set = false;
    bool repeat_penalty_set = false;
    bool presence_penalty_set = false;
    bool frequency_penalty_set = false;
    bool no_repeat_ngram_set = false;
    int dump_layer = -1;
    std::string dump_dir = "debug";
};

void print_usage(const char* prog) {
    std::cout << "Ember - Qwen3 CUDA Inference Engine\n\n";
    std::cout << "Usage: " << prog << " -m <model_path> [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -m, --model PATH        Model directory (with safetensors "
                 "and config.json)\n\n";
    std::cout << "GPU Options:\n";
    std::cout << "  --devices 0,1           GPU devices to use (default: 0)\n";
    std::cout << "  --memory-fraction F     Max memory fraction per GPU "
                 "(default: 0.9)\n\n";
    std::cout << "Inference Options:\n";
    std::cout << "  -c, --ctx-size N        Context length (default: 2048)\n";
    std::cout << "  -n, --n-predict N       Number of tokens to generate "
                 "(default: 128)\n";
    std::cout << "  --temp F                Temperature (default: 0.7)\n";
    std::cout << "  --top-p F               Top-P (default: 0.9)\n";
    std::cout << "  --top-k N               Top-K (default: 40)\n\n";
    std::cout
        << "  --repeat-penalty F      Repetition penalty (default: 1.0)\n\n";
    std::cout << "  --presence-penalty F    Presence penalty (default: 0.0)\n";
    std::cout << "  --frequency-penalty F   Frequency penalty (default: 0.0)\n";
    std::cout
        << "  --no-repeat-ngram N     No-repeat ngram size (default: 0)\n\n";
    std::cout << "Input:\n";
    std::cout << "  -p, --prompt TEXT       Input prompt\n";
    std::cout << "  -i, --interactive       Interactive mode\n\n";
    std::cout << "Debug:\n";
    std::cout << "  -v, --verbose           Verbose output\n";
    std::cout << "  --check                 Correctness check mode\n";
    std::cout
        << "  --dump-layer N          Dump hidden state for layer N (check "
           "mode)\n";
    std::cout << "  --dump-dir PATH         Dump directory (check mode)\n";
    std::cout << "  -h, --help              Show this help\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                std::cerr << "Missing model path\n";
                return false;
            }
            args.model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                std::cerr << "Missing prompt\n";
                return false;
            }
            args.prompt = argv[i];
        } else if (arg == "--devices") {
            if (++i >= argc) {
                std::cerr << "Missing devices\n";
                return false;
            }
            args.devices.clear();
            std::stringstream ss(argv[i]);
            std::string token;
            while (std::getline(ss, token, ',')) {
                args.devices.push_back(std::stoi(token));
            }
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i >= argc) {
                std::cerr << "Missing ctx-size\n";
                return false;
            }
            args.ctx_size = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--n-predict") {
            if (++i >= argc) {
                std::cerr << "Missing n-predict\n";
                return false;
            }
            args.n_predict = std::stoi(argv[i]);
        } else if (arg == "--temp") {
            if (++i >= argc) {
                std::cerr << "Missing temperature\n";
                return false;
            }
            args.temperature = std::stof(argv[i]);
            args.temperature_set = true;
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                std::cerr << "Missing top-p\n";
                return false;
            }
            args.top_p = std::stof(argv[i]);
            args.top_p_set = true;
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                std::cerr << "Missing top-k\n";
                return false;
            }
            args.top_k = std::stoi(argv[i]);
            args.top_k_set = true;
        } else if (arg == "--repeat-penalty") {
            if (++i >= argc) {
                std::cerr << "Missing repeat-penalty\n";
                return false;
            }
            args.repeat_penalty = std::stof(argv[i]);
            args.repeat_penalty_set = true;
        } else if (arg == "--presence-penalty") {
            if (++i >= argc) {
                std::cerr << "Missing presence-penalty\n";
                return false;
            }
            args.presence_penalty = std::stof(argv[i]);
            args.presence_penalty_set = true;
        } else if (arg == "--frequency-penalty") {
            if (++i >= argc) {
                std::cerr << "Missing frequency-penalty\n";
                return false;
            }
            args.frequency_penalty = std::stof(argv[i]);
            args.frequency_penalty_set = true;
        } else if (arg == "--no-repeat-ngram") {
            if (++i >= argc) {
                std::cerr << "Missing no-repeat-ngram\n";
                return false;
            }
            args.no_repeat_ngram = std::stoi(argv[i]);
            args.no_repeat_ngram_set = true;
        } else if (arg == "--memory-fraction") {
            if (++i >= argc) {
                std::cerr << "Missing memory-fraction\n";
                return false;
            }
            args.memory_fraction = std::stof(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "-i" || arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "--check") {
            args.check_mode = true;
        } else if (arg == "--dump-layer") {
            if (++i >= argc) {
                std::cerr << "Missing dump-layer\n";
                return false;
            }
            args.dump_layer = std::stoi(argv[i]);
        } else if (arg == "--dump-dir") {
            if (++i >= argc) {
                std::cerr << "Missing dump-dir\n";
                return false;
            }
            args.dump_dir = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (args.model_path.empty()) {
        std::cerr << "Error: Model path is required (-m)\n";
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    ember_banner();

    int num_gpus = ember::cuda::get_device_count();
}