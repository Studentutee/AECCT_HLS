#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "RefModel.h"
#include "input_y_step0.h"
#include "output_logits_step0.h"
#include "output_x_pred_step0.h"

namespace {

static void fail(const char* msg) {
    std::printf("ERROR: %s\n", msg);
    std::exit(1);
}

static int parse_int(const char* s, const char* name) {
    if (s == nullptr) {
        std::printf("ERROR: missing %s\n", name);
        std::exit(1);
    }
    char* endp = nullptr;
    const long v = std::strtol(s, &endp, 10);
    if (endp == s || *endp != '\0') {
        std::printf("ERROR: invalid %s: %s\n", name, s);
        std::exit(1);
    }
    return static_cast<int>(v);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("Usage: refmodel_dump_step0 <sample_index> <dump_dir>\n");
        return 1;
    }

    const int sample_index = parse_int(argv[1], "sample_index");
    const std::string dump_dir = argv[2];

    if (trace_input_y_step0_tensor_ndim != 2 ||
        trace_output_logits_step0_tensor_ndim != 2 ||
        trace_output_x_pred_step0_tensor_ndim != 2) {
        fail("trace headers must all be rank-2");
    }

    const int b = trace_input_y_step0_tensor_shape[0];
    const int n = trace_input_y_step0_tensor_shape[1];
    if (trace_output_logits_step0_tensor_shape[0] != b ||
        trace_output_logits_step0_tensor_shape[1] != n ||
        trace_output_x_pred_step0_tensor_shape[0] != b ||
        trace_output_x_pred_step0_tensor_shape[1] != n) {
        fail("trace shape mismatch between input/logits/x_pred");
    }

    if (sample_index < 0 || sample_index >= b) {
        std::printf("ERROR: sample_index out of range: %d (valid [0, %d))\n", sample_index, b);
        return 1;
    }

    std::filesystem::create_directories(dump_dir);

    const std::size_t n_vars = static_cast<std::size_t>(n);
    const std::size_t base = static_cast<std::size_t>(sample_index) * n_vars;

    std::vector<double> input_fp32(n_vars, 0.0);
    std::vector<double> out_logits(n_vars, 0.0);
    std::vector<aecct_ref::bit1_t> out_x_pred(n_vars, aecct_ref::bit1_t(0));
    for (std::size_t i = 0; i < n_vars; ++i) {
        input_fp32[i] = trace_input_y_step0_tensor[base + i];
    }

    aecct_ref::RefModel model;
    const aecct_ref::RefRunConfig cfg = aecct_ref::make_fp32_baseline_run_config();
    model.set_run_config(cfg);

    aecct_ref::RefDumpConfig dump_cfg{};
    dump_cfg.enabled = true;
    dump_cfg.dump_dir = dump_dir.c_str();
    dump_cfg.pattern_index = sample_index;
    model.set_dump_config(dump_cfg);

    aecct_ref::RefModelIO io{};
    io.input_y_fp32 = input_fp32.data();
    io.out_logits = out_logits.data();
    io.out_x_pred = out_x_pred.data();
    io.B = 1;
    io.N = static_cast<int>(n_vars);
    model.infer_step0(io);

    double logits_max_abs = 0.0;
    int xpred_mismatch = 0;
    for (std::size_t i = 0; i < n_vars; ++i) {
        const double golden_logit = trace_output_logits_step0_tensor[base + i];
        const double abs_diff = std::fabs(out_logits[i] - golden_logit);
        if (abs_diff > logits_max_abs) {
            logits_max_abs = abs_diff;
        }
        const int golden_x = (trace_output_x_pred_step0_tensor[base + i] != 0.0) ? 1 : 0;
        const int ref_x = out_x_pred[i].to_int();
        if (golden_x != ref_x) {
            ++xpred_mismatch;
        }
    }

    std::printf(
        "refmodel_dump_step0 sample=%d n=%d dump_dir=%s logits_max_abs=%.9e xpred_mismatch=%d\n",
        sample_index,
        n,
        dump_dir.c_str(),
        logits_max_abs,
        xpred_mismatch);
    return 0;
}

