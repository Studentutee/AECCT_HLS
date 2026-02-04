#include <cstdio>
#include <algorithm>

#include "ac_channel.h"
#include "ac_fixed.h"
#include "ac_int.h"
#include "compare_array_abs.h"

// ===============================
// 你產生的 header 檔
// ===============================
#include "layer0_attn_post_concat_step0.h"
#include "layer0_attn_out_step0.h"

// ===============================
// 你的 DUT header
// ===============================
#include "Layer0.h"

// ===============================
// TB 參數
// ===============================
#ifndef ATOL
#define ATOL (1e-3)  // absolute tolerance
#endif

#ifndef FAIL_FAST
#define FAIL_FAST 0  // 1: mismatch 就停；0: 跑完再統計
#endif

// 若你只想跑單一筆 pattern：把 PATTERN_INDEX 設成 >=0
#ifndef PATTERN_INDEX
#define PATTERN_INDEX (-1)
#endif

int main() {
    std::printf("==== Trace-based TB start (L0_OutProj) ====\n");
    std::printf("ATOL = %.10g\n", (double)ATOL);

    // -------------------------------
    // 0) 基本尺寸（從 trace header 直接拿）
    // -------------------------------
    // TODO: 把下面這些 symbol 名稱換成你 trace header 的實際名稱
    //
    // 假設輸入 tensor shape: [NUM_SAMPLES, N_NODES, D_MODEL] 或 [NUM_SAMPLES, VEC_LEN]
    // 假設輸出 tensor shape: [NUM_SAMPLES, N_NODES, D_MODEL] 或 [NUM_SAMPLES, VEC_LEN]
    //
    // 你在 PreprocEmbedSPE 的 TB 用的是：
    //   trace_input_y_step0_tensor_shape
    //   trace_embed_plus_SPE_step0_tensor_shape
    //
    // 這裡請對應成 OutProj 的：
    //   trace_<in>_shape
    //   trace_<exp>_shape

    constexpr int NUM_SAMPLES = trace_layer0_attn_post_concat_step0_tensor_shape[0];
    constexpr int IN_DIM1     = trace_layer0_attn_post_concat_step0_tensor_shape[1];
    constexpr int IN_DIM2     = trace_layer0_attn_post_concat_step0_tensor_shape[2];

    constexpr int EXP_DIM1    = trace_layer0_attn_out_step0_tensor_shape[1];
    constexpr int EXP_DIM2    = trace_layer0_attn_out_step0_tensor_shape[2];

    constexpr int N_NODES = PreprocEmbedSPE::N_NODES; // 75
    constexpr int D_MODEL = PreprocEmbedSPE::D_MODEL; // 32
    constexpr int VEC_LEN = N_NODES * D_MODEL;        // 2400

    // -------------------------------
    // 1) 建立/初始化 DUT
    // -------------------------------
    L0_OutProj dut;

    // -------------------------------
    // 2) 準備 buffer
    // -------------------------------
    // 這裡直接用 fx_utils::fx_t（因為 DUT 的 channel 型態就是它）
    // 若你的 trace 存的是 double / float，都可以在寫入時轉成 fx_utils::fx_t
    static fx_utils::fx_t in_buf[VEC_LEN];
    static fx_utils::fx_t out_buf[VEC_LEN];

    // -------------------------------
    // 3) 決定要跑哪些 sample
    // -------------------------------
    int s_begin = 0;
    int s_end   = NUM_SAMPLES; // [s_begin, s_end)
    if (PATTERN_INDEX >= 0) {
        if (PATTERN_INDEX >= NUM_SAMPLES) {
            std::printf("ERROR: PATTERN_INDEX=%d out of range (0..%d)\n", PATTERN_INDEX, NUM_SAMPLES - 1);
            return 2;
        }
        s_begin = PATTERN_INDEX;
        s_end   = PATTERN_INDEX + 1;
    }

    // -------------------------------
    // 4) 逐筆餵 input 給 DUT，並與 expected 比對
    // -------------------------------
    int total_mis = 0;

    for (int s = s_begin; s < s_end; ++s) {
        // fresh channels per sample
        ac_channel<fx_utils::fx_t> in_ch;
        ac_channel<fx_utils::fx_t> out_ch;

        // 4.1 取得 input 指標
        // TODO: 把 trace_L0_OUTPROJ_IN_tensor 換成你的實際 tensor 名稱
        const double* in_ptr = &trace_layer0_attn_post_concat_step0_tensor[s * VEC_LEN];

        // 4.2 double/float -> fx_utils::fx_t
        for (int i = 0; i < VEC_LEN; ++i) {
            in_buf[i] = fx_utils::fx_t(in_ptr[i]);
        }

        // Push inputs into channel (order i=0..VEC_LEN-1)
        for (int i = 0; i < VEC_LEN; ++i) {
            in_ch.write(in_buf[i]);
        }

        // 4.3 取得 expected 指標
        // TODO: 把 trace_L0_OUTPROJ_EXP_tensor 換成你的實際 tensor 名稱
        const double* exp_ptr = &trace_layer0_attn_out_step0_tensor[s * VEC_LEN];

        // 4.4 呼叫 DUT
        dut.run(in_ch, out_ch);

        // Pop all outputs from channel into flat buffer
        for (int j = 0; j < VEC_LEN; ++j) {
            out_buf[j] = out_ch.read();
        }

        // 4.5 比對
        char name[64];
        std::snprintf(name, sizeof(name), "L0_OutProj_out(sample=%d)", s);

        int mis = compare_array_abs<double, fx_utils::fx_t>(
            exp_ptr,
            out_buf,
            VEC_LEN,
            (double)ATOL,
            name
        );

        total_mis += mis;

#if FAIL_FAST
        if (mis) break;
#endif
    }

    // -------------------------------
    // 5) Summary
    // -------------------------------
    if (total_mis == 0) {
        std::printf("==== PASS: all compared values within ATOL ====\n");
        return 0;
    } else {
        std::printf("==== FAIL: total mismatches = %d ====\n", total_mis);
        return 1;
    }
}
