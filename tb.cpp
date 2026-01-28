#include <cstdio>
#include <algorithm>
#include "ac_channel.h"
#include "ac_fixed.h"
#include "ac_int.h"
#include "compare_array_abs.h"

// ===============================
// 你產生的 header 檔
// ===============================
// 權重（若你的 DUT 需要在 TB 端初始化/載入權重就用得到）
#include "weights.h"

// Trace (.pt) 轉出的 .h：
// - trace_input_y_step0_tensor            : [NUM_SAMPLES, Y_LEN]
// - trace_embed_node_embed_step0_tensor   : [NUM_SAMPLES, OUT_H, OUT_W]
#include "input_y_step0.h"
#include "embed_plus_SPE_step0.h"

// ===============================
// 你的 DUT header
// ===============================
// 需求：DUT 端至少要提供一個可以「吃進 y、吐出 out」的呼叫介面。
// 下面 TB 先假設有：
//   void Dut::run(const float* y_in, float* out);
// 若你的介面是 ac_channel / Connections，請看本文後面的「介面改成通道」註解區，照著改即可。
#include "PreprocEmbedSPE.h"

// ===============================
// TB 參數
// ===============================
#ifndef ATOL
#define ATOL (1e-3)  // absolute tolerance
#endif

#ifndef FAIL_FAST
#define FAIL_FAST 0  // 1: 一旦 mismatch 就停；0: 跑完再統計
#endif

// 若你只想跑單一筆 pattern：把 PATTERN_INDEX 設成 >=0
// 例如：-DPATTERN_INDEX=17
#ifndef PATTERN_INDEX
#define PATTERN_INDEX (-1)
#endif

int main() {
    std::printf("==== Trace-based TB start ====\n");
    std::printf("ATOL = %.10g\n", (double)ATOL);

    // -------------------------------
    // 0) 基本尺寸（從 trace header 直接拿）
    // -------------------------------
    constexpr int NUM_SAMPLES = trace_input_y_step0_tensor_shape[0];
    constexpr int Y_LEN = trace_input_y_step0_tensor_shape[1];

    constexpr int OUT_H = trace_embed_plus_SPE_step0_tensor_shape[1]; // 75
    constexpr int OUT_W = trace_embed_plus_SPE_step0_tensor_shape[2]; // 32
    constexpr int OUT_LEN = OUT_H * OUT_W;                             // 2400

    static_assert(trace_input_y_step0_tensor_shape[0] == trace_embed_plus_SPE_step0_tensor_shape[0],
        "input_y and expected_out must have same NUM_SAMPLES");

    // -------------------------------
    // 1) 建立/初始化 DUT
    // -------------------------------
    // TODO: 依你的 DUT 需求調整
    // 例：
    //   Dut dut(weights...);
    // 或：
    //   Dut dut; dut.load_weights(...);
    PreprocEmbedSPE dut;

    // -------------------------------
    // 2) 準備 buffer
    // -------------------------------
    // 注意：這裡用 float 是為了先把「餵 y」這件事做通。
    // 之後你要接固定點（ac_fixed）再換型態即可。
	static ac_fixed<32, 16, true> y_buf[Y_LEN]; // Q3.5
    static ac_fixed<32, 16, true> out_buf[OUT_LEN]; // Q3.5

    // -------------------------------
    // 3) 決定要跑哪些 sample
    // -------------------------------
    int s_begin = 0;
    int s_end = 100;// NUM_SAMPLES; // [s_begin, s_end)
    if (PATTERN_INDEX >= 0) {
        if (PATTERN_INDEX >= NUM_SAMPLES) {
            std::printf("ERROR: PATTERN_INDEX=%d out of range (0..%d)\n", PATTERN_INDEX, NUM_SAMPLES - 1);
            return 2;
        }
        s_begin = PATTERN_INDEX;
        s_end = PATTERN_INDEX + 1;
    }

    // -------------------------------
    // 4) 逐筆餵 y 給 DUT，並與 expected 比對
    // -------------------------------
    int total_mis = 0;

    for (int s = s_begin; s < s_end; ++s) {
        // 4.1 取得 y 指標：flat array -> (s, :)
        const double* y_ptr = &trace_input_y_step0_tensor[s * Y_LEN];

        // 4.2 float -> ac_fixed (會量化/截位，誤差大小與你的 ac_fixed 設定有關)
		for (int i = 0; i < Y_LEN; ++i) { //不同型態矩陣不能直接 memcpy, 只能一個一個轉
			y_buf[i] = ac_fixed<32, 16, true, AC_RND_CONV, AC_SAT_SYM>(y_ptr[i]); //data quantization
        }
        if (s == 1020) {
            std::printf("y_ptr[0]=%.9g  y_buf[0]=%.9g\n", y_ptr[0], y_buf[0].to_double());
        }
        // 4.3 取得 expected out 指標：flat array -> (s, :, :)
        const double* exp_ptr = &trace_embed_plus_SPE_step0_tensor[s * OUT_LEN];

        // 4.4 呼叫 DUT
        // 目前假設：dut.run(const float* y_in, float* out)
        dut.run(y_buf, out_buf);

        // 4.5 比對
        char name[64];
		// 產生 tensor name 字串
        std::snprintf(name, sizeof(name), "embed_plus_SPE_out(sample=%d)", s);

		int mis = compare_array_abs<double, ac_fixed<32, 16, true>>( // Q3.5
            exp_ptr,
            out_buf,
            OUT_LEN,
            (double)ATOL,
            name);

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
    }
    else {
        std::printf("==== FAIL: total mismatches = %d ====\n", total_mis);
        return 1;
    }
}