// P00-011AD: Top-managed Q path mainline implementation proof (local-only).

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "Top.h"
#include "blocks/AttnPhaseATopManagedQ.h"
#include "blocks/TernaryLiveQkvLeafKernel.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "weights.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#define AECCT_HAS_SCVERIFY 1
#else
#define AECCT_HAS_SCVERIFY 0
#endif

#if !AECCT_HAS_SCVERIFY
#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif
#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif
#endif

namespace {

static const uint32_t kTokenCount = 2u;
static const uint32_t kDTileCount = 1u;
static const uint32_t kTileWords = (uint32_t)aecct::ATTN_TOP_MANAGED_TILE_WORDS;

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static uint32_t encode_ternary_code(double w) {
    if (w > 0.5) {
        return (uint32_t)TERNARY_CODE_POS;
    }
    if (w < -0.5) {
        return (uint32_t)TERNARY_CODE_NEG;
    }
    return (uint32_t)TERNARY_CODE_ZERO;
}

static bool matrix_weight_at(uint32_t matrix_id, uint32_t elem_idx, double& out_w) {
    if (matrix_id == (uint32_t)QLM_L0_WQ) {
        out_w = w_decoder_layers_0_self_attn_linears_0_weight[elem_idx];
        return true;
    }
    return false;
}

static bool matrix_inv_sw(uint32_t matrix_id, double& out_inv_sw) {
    if (matrix_id == (uint32_t)QLM_L0_WQ) {
        out_inv_sw = w_decoder_layers_0_self_attn_linears_0_s_w[0];
        return true;
    }
    return false;
}

static bool build_payload_for_matrix(
    uint32_t matrix_id,
    const QuantLinearMeta& meta,
    std::vector<aecct::u32_t>& out_payload) {
    out_payload.assign(meta.payload_words_2b, (aecct::u32_t)0u);
    for (uint32_t out = 0u; out < meta.rows; ++out) {
        for (uint32_t in = 0u; in < meta.cols; ++in) {
            const uint32_t elem_idx = out * meta.cols + in;
            const uint32_t word_idx = (elem_idx >> 4);
            const uint32_t slot = (elem_idx & 15u);
            if (word_idx >= meta.payload_words_2b) {
                return false;
            }

            double w = 0.0;
            if (!matrix_weight_at(matrix_id, elem_idx, w)) {
                return false;
            }
            const uint32_t code = encode_ternary_code(w);
            if (code == (uint32_t)TERNARY_CODE_RSVD) {
                return false;
            }
            out_payload[word_idx] = (aecct::u32_t)((uint32_t)out_payload[word_idx].to_uint() |
                                                   ((code & 0x3u) << (slot * 2u)));
        }
    }
    return true;
}

class TbP11ad {
public:
    TbP11ad() : mainline_q_path_taken_(false), fallback_taken_(true), expected_q_sx_bits_(0u) {}

    int run_all() {
        init();
        if (!prepare_payload_and_expected()) {
            std::printf("[p11ad][FAIL] payload/expected preparation failed\n");
            return 1;
        }
        if (!run_legacy_work_unit_split_probe()) {
            return 1;
        }
        if (!run_design_mainline_probe()) {
            return 1;
        }
        if (!validate_expected_compare()) {
            return 1;
        }
        if (!validate_target_span_write()) {
            return 1;
        }
        if (!validate_no_spurious_write_and_source_preservation()) {
            return 1;
        }
        std::printf("PASS: tb_q_path_impl_p11ad\n");
        return 0;
    }

private:
    bool mainline_q_path_taken_;
    bool fallback_taken_;
    aecct::u32_t expected_q_sx_bits_;

    std::vector<aecct::u32_t> sram_;
    std::vector<aecct::u32_t> sram_before_;
    std::vector<aecct::u32_t> wq_payload_;
    aecct::u32_t wq_inv_sw_bits_;
    std::vector<aecct::u32_t> expected_q_[kTokenCount];
    std::vector<aecct::u32_t> expected_q_act_q_[kTokenCount];

    static uint32_t x_row_base(uint32_t token) {
        return (uint32_t)aecct::LN_X_OUT_BASE_WORD + token * kTileWords;
    }

    static uint32_t q_row_base(const aecct::AttnScratch& sc, uint32_t token) {
        return (uint32_t)sc.q_base_word.to_uint() + token * kTileWords;
    }

    static uint32_t q_act_q_row_base(const aecct::AttnScratch& sc, uint32_t token) {
        return (uint32_t)sc.q_act_q_base_word.to_uint() + token * kTileWords;
    }

    void init() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t base = x_row_base(t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                const int32_t v = (int32_t)((t + 1u) * 23u + (i * 7u)) - 41;
                const float f = ((float)v) * 0.03125f;
                sram_[base + i] = (aecct::u32_t)f32_to_bits(f);
            }
        }
        sram_before_ = sram_;
    }

    bool prepare_payload_and_expected() {
        const QuantLinearMeta wq_meta = aecct::ternary_linear_live_l0_wq_meta();
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WQ, wq_meta, wq_payload_)) {
            return false;
        }

        double wq_inv_sw = 0.0;
        if (!matrix_inv_sw((uint32_t)QLM_L0_WQ, wq_inv_sw)) {
            return false;
        }
        wq_inv_sw_bits_ = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wq_inv_sw);
        expected_q_sx_bits_ = wq_inv_sw_bits_;

        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            expected_q_[t].assign(kTileWords, (aecct::u32_t)0u);
            expected_q_act_q_[t].assign(kTileWords, (aecct::u32_t)0u);

            aecct::u32_t x_row[kTileWords];
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                x_row[i] = sram_[x_row_base(t) + i];
            }

            aecct::u32_t out_inv_sw_bits = (aecct::u32_t)0u;
            if (!aecct::ternary_live_l0_wq_materialize_row_kernel_split(
                    x_row,
                    wq_payload_.data(),
                    wq_inv_sw_bits_,
                    expected_q_[t].data(),
                    expected_q_act_q_[t].data(),
                    out_inv_sw_bits)) {
                return false;
            }
            if ((uint32_t)out_inv_sw_bits.to_uint() != (uint32_t)wq_inv_sw_bits_.to_uint()) {
                return false;
            }
        }
        return true;
    }

    bool run_legacy_work_unit_split_probe() {
        std::vector<aecct::u32_t> probe_sram = sram_before_;
        const aecct::LayerScratch layer_sc =
            aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const aecct::AttnScratch sc = layer_sc.attn;

        aecct::attn_q_x_pkt_ch_t x_ch;
        aecct::attn_q_wq_pkt_ch_t wq_ch;
        aecct::attn_q_pkt_ch_t q_ch;

        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            if (!aecct::attn_top_emit_phasea_q_work_unit(
                    probe_sram,
                    (aecct::u32_t)x_row_base(t),
                    (aecct::u32_t)t,
                    (aecct::u32_t)0u,
                    x_ch,
                    wq_ch)) {
                std::printf("[p11ad][FAIL] legacy split emit failed token=%u\n", (unsigned)t);
                return false;
            }

            if (!aecct::attn_block_phasea_q_consume_emit(
                    x_ch,
                    wq_ch,
                    q_ch,
                    wq_payload_.data(),
                    wq_inv_sw_bits_)) {
                std::printf("[p11ad][FAIL] legacy split consume failed token=%u\n", (unsigned)t);
                return false;
            }

            if (!aecct::attn_top_writeback_phasea_q_work_unit(
                    probe_sram,
                    (aecct::u32_t)q_row_base(sc, t),
                    (aecct::u32_t)q_act_q_row_base(sc, t),
                    (aecct::u32_t)sc.q_sx_base_word.to_uint(),
                    (aecct::u32_t)t,
                    (aecct::u32_t)0u,
                    q_ch)) {
                std::printf("[p11ad][FAIL] legacy split writeback failed token=%u\n", (unsigned)t);
                return false;
            }

            const uint32_t q_base = q_row_base(sc, t);
            const uint32_t q_act_q_base = q_act_q_row_base(sc, t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                const uint32_t got_q = (uint32_t)probe_sram[q_base + i].to_uint();
                const uint32_t exp_q = (uint32_t)expected_q_[t][i].to_uint();
                if (got_q != exp_q) {
                    std::printf("[p11ad][FAIL] legacy split Q mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                                (unsigned)t, (unsigned)i, (unsigned)got_q, (unsigned)exp_q);
                    return false;
                }

                const uint32_t got_q_act_q = (uint32_t)probe_sram[q_act_q_base + i].to_uint();
                const uint32_t exp_q_act_q = (uint32_t)expected_q_act_q_[t][i].to_uint();
                if (got_q_act_q != exp_q_act_q) {
                    std::printf("[p11ad][FAIL] legacy split Q_act_q mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                                (unsigned)t, (unsigned)i, (unsigned)got_q_act_q, (unsigned)exp_q_act_q);
                    return false;
                }
            }
        }

        const uint32_t got_q_sx = (uint32_t)probe_sram[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint();
        const uint32_t exp_q_sx = (uint32_t)expected_q_sx_bits_.to_uint();
        if (got_q_sx != exp_q_sx) {
            std::printf("[p11ad][FAIL] legacy split Q_sx mismatch got=0x%08X exp=0x%08X\n",
                        (unsigned)got_q_sx,
                        (unsigned)exp_q_sx);
            return false;
        }

        std::printf("LEGACY_WORK_UNIT_SPLIT_PATH PASS\n");
        return true;
    }

    bool run_design_mainline_probe() {
        const QuantLinearMeta wq_meta = aecct::ternary_linear_live_l0_wq_meta();
        const ParamMeta wq_payload_meta = kParamMeta[wq_meta.weight_param_id];
        const ParamMeta wq_inv_meta = kParamMeta[wq_meta.inv_sw_param_id];
        const uint32_t param_base = (uint32_t)sram_map::W_REGION_BASE;

        for (uint32_t i = 0u; i < wq_meta.payload_words_2b; ++i) {
            sram_[param_base + wq_payload_meta.offset_w + i] = wq_payload_[i];
        }
        sram_[param_base + wq_inv_meta.offset_w] = wq_inv_sw_bits_;
        sram_before_ = sram_;

        aecct::CfgRegs cfg;
        cfg.d_model = (aecct::u32_t)kTileWords;
        cfg.n_heads = (aecct::u32_t)1u;
        cfg.d_ffn = (aecct::u32_t)kTileWords;
        cfg.n_layers = (aecct::u32_t)1u;
        const aecct::LayerScratch top_sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const aecct::LayerParamBase top_pb =
            aecct::make_layer_param_base((aecct::u32_t)param_base, (aecct::u32_t)0u);

        fallback_taken_ = true;
        mainline_q_path_taken_ = aecct::run_p11ad_layer0_top_managed_q(
            sram_.data(),
            cfg,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            top_sc,
            top_pb,
            fallback_taken_);
        std::printf("fallback_taken = %s\n", fallback_taken_ ? "true" : "false");
        if (!mainline_q_path_taken_) {
            std::printf("[p11ad][FAIL] Top mainline Q path was not taken\n");
            return false;
        }
        if (fallback_taken_) {
            std::printf("[p11ad][FAIL] fallback path was taken in Top mainline Q probe\n");
            return false;
        }

        std::printf("Q_PATH_MAINLINE PASS\n");
        return true;
    }

    bool validate_expected_compare() {
        const aecct::LayerScratch layer_sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const aecct::AttnScratch sc = layer_sc.attn;
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t q_base = q_row_base(sc, t);
            const uint32_t q_act_q_base = q_act_q_row_base(sc, t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                const uint32_t got_q = (uint32_t)sram_[q_base + i].to_uint();
                const uint32_t exp_q = (uint32_t)expected_q_[t][i].to_uint();
                if (got_q != exp_q) {
                    std::printf("[p11ad][FAIL] Q compare mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                                (unsigned)t, (unsigned)i, (unsigned)got_q, (unsigned)exp_q);
                    return false;
                }

                const uint32_t got_q_act_q = (uint32_t)sram_[q_act_q_base + i].to_uint();
                const uint32_t exp_q_act_q = (uint32_t)expected_q_act_q_[t][i].to_uint();
                if (got_q_act_q != exp_q_act_q) {
                    std::printf("[p11ad][FAIL] Q_act_q compare mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                                (unsigned)t, (unsigned)i, (unsigned)got_q_act_q, (unsigned)exp_q_act_q);
                    return false;
                }
            }
        }

        const uint32_t got_q_sx = (uint32_t)sram_[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint();
        const uint32_t exp_q_sx = (uint32_t)expected_q_sx_bits_.to_uint();
        if (got_q_sx != exp_q_sx) {
            std::printf("[p11ad][FAIL] Q_sx mismatch got=0x%08X exp=0x%08X\n",
                        (unsigned)got_q_sx,
                        (unsigned)exp_q_sx);
            return false;
        }
        std::printf("Q_EXPECTED_COMPARE PASS\n");
        return true;
    }

    bool validate_target_span_write() {
        const aecct::LayerScratch layer_sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const aecct::AttnScratch sc = layer_sc.attn;
        uint32_t changed_q = 0u;
        uint32_t changed_q_act_q = 0u;
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t q_base = q_row_base(sc, t);
            const uint32_t q_act_q_base = q_act_q_row_base(sc, t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                if ((uint32_t)sram_[q_base + i].to_uint() != (uint32_t)sram_before_[q_base + i].to_uint()) {
                    ++changed_q;
                }
                if ((uint32_t)sram_[q_act_q_base + i].to_uint() != (uint32_t)sram_before_[q_act_q_base + i].to_uint()) {
                    ++changed_q_act_q;
                }
            }
        }
        const uint32_t q_sx_addr = (uint32_t)sc.q_sx_base_word.to_uint();
        const bool changed_q_sx =
            ((uint32_t)sram_[q_sx_addr].to_uint() != (uint32_t)sram_before_[q_sx_addr].to_uint());

        if (changed_q == 0u && changed_q_act_q == 0u && !changed_q_sx) {
            std::printf("[p11ad][FAIL] target spans were not written\n");
            return false;
        }

        std::printf("[p11ad][TARGET_SPAN_WRITE][PASS] changed_q=%u changed_q_act_q=%u changed_q_sx=%s\n",
                    (unsigned)changed_q,
                    (unsigned)changed_q_act_q,
                    changed_q_sx ? "true" : "false");
        std::printf("Q_TARGET_SPAN_WRITE PASS\n");
        return true;
    }

    bool validate_no_spurious_write_and_source_preservation() {
        const aecct::LayerScratch layer_sc = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);
        const aecct::AttnScratch sc = layer_sc.attn;
        std::vector<uint8_t> allowed_write((uint32_t)sram_.size(), 0u);
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t q_base = q_row_base(sc, t);
            const uint32_t q_act_q_base = q_act_q_row_base(sc, t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                allowed_write[q_base + i] = 1u;
                allowed_write[q_act_q_base + i] = 1u;
            }
            const uint32_t x_base = x_row_base(t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                if ((uint32_t)sram_[x_base + i].to_uint() != (uint32_t)sram_before_[x_base + i].to_uint()) {
                    std::printf("[p11ad][FAIL] source preservation mismatch token=%u idx=%u\n",
                                (unsigned)t, (unsigned)i);
                    return false;
                }
            }
        }
        allowed_write[(uint32_t)sc.q_sx_base_word.to_uint()] = 1u;

        for (uint32_t i = 0u; i < (uint32_t)sram_.size(); ++i) {
            const uint32_t before = (uint32_t)sram_before_[i].to_uint();
            const uint32_t after = (uint32_t)sram_[i].to_uint();
            if (before != after && allowed_write[i] == 0u) {
                std::printf("[p11ad][FAIL] no-spurious-write mismatch addr=%u before=0x%08X after=0x%08X\n",
                            (unsigned)i, (unsigned)before, (unsigned)after);
                return false;
            }
        }

        std::printf("NO_SPURIOUS_WRITE PASS\n");
        std::printf("SOURCE_PRESERVATION PASS\n");
        std::printf("MAINLINE_Q_PATH_TAKEN PASS\n");
        std::printf("FALLBACK_NOT_TAKEN PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11ad tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
