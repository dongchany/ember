#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/cache_policy.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

ember::CachePolicyType parse_policy(const std::string& s) {
    if (s == "naive") return ember::CachePolicyType::NAIVE;
    if (s == "update_locality") return ember::CachePolicyType::UPDATE_LOCALITY;
    if (s == "periodic_refresh") return ember::CachePolicyType::PERIODIC_REFRESH;
    die("unknown --policy: " + s);
}

}  // namespace

int main(int argc, char** argv) {
    std::string policy = "naive";
    int num_layers = 36;
    int rounds = 30;
    int freeze_layers = 18;
    int periodic_refresh_k = 10;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Cache Policy Simulator\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " [options]\n\n"
                << "Options:\n"
                << "  --policy STR          naive|update_locality|periodic_refresh (default: naive)\n"
                << "  --num-layers N        model layers (default: 36)\n"
                << "  --rounds N            update rounds (default: 30)\n"
                << "  --freeze-layers N     reused prefix layers for locality (default: 18)\n"
                << "  --periodic-refresh-k N (default: 10)\n"
                << "  --csv PATH            output csv path\n";
            return 0;
        } else if (arg == "--policy") {
            policy = need("--policy");
        } else if (arg == "--num-layers") {
            num_layers = std::stoi(need("--num-layers"));
        } else if (arg == "--rounds") {
            rounds = std::stoi(need("--rounds"));
        } else if (arg == "--freeze-layers") {
            freeze_layers = std::stoi(need("--freeze-layers"));
        } else if (arg == "--periodic-refresh-k") {
            periodic_refresh_k = std::stoi(need("--periodic-refresh-k"));
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (num_layers <= 0) die("--num-layers must be > 0");
    if (rounds <= 0) die("--rounds must be > 0");
    if (freeze_layers < 0 || freeze_layers > num_layers) die("--freeze-layers out of range");
    if (periodic_refresh_k < 0) die("--periodic-refresh-k must be >= 0");

    ember::CachePolicyConfig cfg;
    cfg.type = parse_policy(policy);
    cfg.total_layers = num_layers;
    cfg.freeze_prefix_layers = freeze_layers;
    cfg.periodic_refresh_k = periodic_refresh_k;

    ember::CachePolicyEngine engine(cfg);

    std::vector<std::string> lines;
    lines.push_back("round,policy,num_layers,freeze_layers,periodic_refresh_k,full_refresh,recompute_layers,reused_layers,recompute_ratio");

    for (int r = 1; r <= rounds; ++r) {
        ember::CachePolicyDecision d = engine.on_policy_update();
        std::ostringstream row;
        row << d.update_step << ","
            << policy << ","
            << num_layers << ","
            << freeze_layers << ","
            << periodic_refresh_k << ","
            << (d.full_refresh ? 1 : 0) << ","
            << d.recompute_layers << ","
            << d.reused_layers << ","
            << std::fixed << std::setprecision(6) << d.recompute_ratio;
        lines.push_back(row.str());
    }

    std::ostringstream summary;
    const auto& st = engine.stats();
    summary << "# summary"
            << "\npolicy=" << policy
            << "\nupdates=" << st.updates
            << "\nfull_refreshes=" << st.full_refreshes
            << "\navg_recompute_ratio=" << std::fixed << std::setprecision(6) << st.avg_recompute_ratio
            << "\ntotal_recompute_layers=" << st.total_recompute_layers
            << "\ntotal_reused_layers=" << st.total_reused_layers
            << "\n";

    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (!out.is_open()) die("failed to open csv path: " + csv_path);
        for (const auto& ln : lines) out << ln << "\n";
        out << summary.str();
    } else {
        for (const auto& ln : lines) std::cout << ln << "\n";
        std::cout << summary.str();
    }

    return 0;
}

