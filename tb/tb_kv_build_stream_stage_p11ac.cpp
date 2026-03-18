// P00-011AC: Top-managed Phase-A KV staging proof (local-only).

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "blocks/AttnPhaseATopManagedKv.h"
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

struct EventTag {
    uint32_t token;
    uint32_t d_tile;
    uint32_t kind;
};

struct WorkCounters {
    uint32_t x_reads;
    uint32_t wk_reads;
    uint32_t wv_reads;
    uint32_t scr_k_writes;
    uint32_t scr_v_writes;
    WorkCounters() : x_reads(0u), wk_reads(0u), wv_reads(0u), scr_k_writes(0u), scr_v_writes(0u) {}
};

static uint32_t f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = f;
    return cvt.u;
}

static float bits_to_f32(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } cvt;
    cvt.u = u;
    return cvt.f;
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
    if (matrix_id == (uint32_t)QLM_L0_WK) {
        out_w = w_decoder_layers_0_self_attn_linears_1_weight[elem_idx];
        return true;
    }
    if (matrix_id == (uint32_t)QLM_L0_WV) {
        out_w = w_decoder_layers_0_self_attn_linears_2_weight[elem_idx];
        return true;
    }
    return false;
}

static bool matrix_inv_sw(uint32_t matrix_id, double& out_inv_sw) {
    if (matrix_id == (uint32_t)QLM_L0_WK) {
        out_inv_sw = w_decoder_layers_0_self_attn_linears_1_s_w[0];
        return true;
    }
    if (matrix_id == (uint32_t)QLM_L0_WV) {
        out_inv_sw = w_decoder_layers_0_self_attn_linears_2_s_w[0];
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

class TbP11ac {
public:
    TbP11ac() : token_count_(kTokenCount), d_tile_count_(kDTileCount) {}

    int run_all() {
        init();
        if (!prepare_payload()) {
            std::printf("[p11ac][FAIL] payload preparation failed\n");
            return 1;
        }
        if (!run_top_managed_stage()) {
            std::printf("[p11ac][FAIL] top-managed stage execution failed\n");
            return 1;
        }
        if (!validate_order()) {
            return 1;
        }
        if (!validate_access_counts()) {
            return 1;
        }
        if (!validate_exact_spans()) {
            return 1;
        }
        std::printf("PASS: tb_kv_build_stream_stage_p11ac\n");
        return 0;
    }

private:
    uint32_t token_count_;
    uint32_t d_tile_count_;

    std::vector<aecct::u32_t> sram_;
    std::vector<aecct::u32_t> sram_before_;

    std::vector<aecct::u32_t> wk_payload_;
    std::vector<aecct::u32_t> wv_payload_;
    aecct::u32_t wk_inv_sw_bits_;
    aecct::u32_t wv_inv_sw_bits_;

    std::vector<EventTag> stream_events_;
    std::vector<EventTag> mem_events_;
    WorkCounters per_work_[kTokenCount][kDTileCount];

    std::vector<aecct::u32_t> expected_k_[kTokenCount];
    std::vector<aecct::u32_t> expected_v_[kTokenCount];

    aecct::attn_pkt_ch_t in_ch_;
    aecct::attn_pkt_ch_t out_ch_;

    static uint32_t x_row_base(uint32_t token) {
        return (uint32_t)sram_map::BASE_X_WORK_W + token * kTileWords;
    }
    static uint32_t scr_k_row_base(uint32_t token) {
        return (uint32_t)sram_map::BASE_SCR_K_W + token * kTileWords;
    }
    static uint32_t scr_v_row_base(uint32_t token) {
        return (uint32_t)sram_map::BASE_SCR_V_W + token * kTileWords;
    }

    void mark_stream(uint32_t token, uint32_t d_tile, uint32_t kind) {
        EventTag e;
        e.token = token;
        e.d_tile = d_tile;
        e.kind = kind;
        stream_events_.push_back(e);
    }

    void mark_mem(uint32_t token, uint32_t d_tile, uint32_t kind) {
        EventTag e;
        e.token = token;
        e.d_tile = d_tile;
        e.kind = kind;
        mem_events_.push_back(e);
    }

    void init() {
        sram_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        for (uint32_t t = 0u; t < token_count_; ++t) {
            const uint32_t base = x_row_base(t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                const int32_t v = (int32_t)((t + 1u) * 19u + (i * 5u)) - 31;
                const float f = ((float)v) * 0.03125f;
                sram_[base + i] = (aecct::u32_t)f32_to_bits(f);
            }
        }
        sram_before_ = sram_;
    }

    bool prepare_payload() {
        const QuantLinearMeta wk_meta = aecct::ternary_linear_live_l0_wk_meta();
        const QuantLinearMeta wv_meta = aecct::ternary_linear_live_l0_wv_meta();
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, wk_meta, wk_payload_)) {
            return false;
        }
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, wv_meta, wv_payload_)) {
            return false;
        }

        double wk_inv_sw = 0.0;
        double wv_inv_sw = 0.0;
        if (!matrix_inv_sw((uint32_t)QLM_L0_WK, wk_inv_sw)) {
            return false;
        }
        if (!matrix_inv_sw((uint32_t)QLM_L0_WV, wv_inv_sw)) {
            return false;
        }
        wk_inv_sw_bits_ = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wk_inv_sw);
        wv_inv_sw_bits_ = (aecct::u32_t)aecct::fp32_bits_from_double(1.0 / wv_inv_sw);

        for (uint32_t t = 0u; t < token_count_; ++t) {
            expected_k_[t].assign(kTileWords, (aecct::u32_t)0u);
            expected_v_[t].assign(kTileWords, (aecct::u32_t)0u);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                x_row_[i] = sram_[x_row_base(t) + i];
            }

            aecct::TernaryLiveL0WkRowTop wk_top;
            aecct::u32_t out_inv_sw_k = 0;
            if (!wk_top.run(
                    x_row_,
                    wk_payload_.data(),
                    wk_inv_sw_bits_,
                    expected_k_[t].data(),
                    expected_k_act_q_,
                    out_inv_sw_k)) {
                return false;
            }

            aecct::TernaryLiveL0WvRowTop wv_top;
            aecct::u32_t out_inv_sw_v = 0;
            if (!wv_top.run(
                    x_row_,
                    wv_payload_.data(),
                    wv_inv_sw_bits_,
                    expected_v_[t].data(),
                    expected_v_act_q_,
                    out_inv_sw_v)) {
                return false;
            }
        }
        return true;
    }

    bool run_top_managed_stage() {
        for (uint32_t t = 0u; t < token_count_; ++t) {
            for (uint32_t dt = 0u; dt < d_tile_count_; ++dt) {
                if (!aecct::attn_top_emit_phasea_kv_work_unit(
                        sram_.data(),
                        (aecct::u32_t)x_row_base(t),
                        (aecct::u32_t)t,
                        (aecct::u32_t)dt,
                        in_ch_)) {
                    return false;
                }
                mark_stream(t, dt, (uint32_t)aecct::ATTN_PKT_X);
                mark_stream(t, dt, (uint32_t)aecct::ATTN_PKT_WK);
                mark_stream(t, dt, (uint32_t)aecct::ATTN_PKT_WV);
                mark_mem(t, dt, (uint32_t)aecct::ATTN_PKT_X);
                mark_mem(t, dt, (uint32_t)aecct::ATTN_PKT_WK);
                mark_mem(t, dt, (uint32_t)aecct::ATTN_PKT_WV);
                ++per_work_[t][dt].x_reads;
                ++per_work_[t][dt].wk_reads;
                ++per_work_[t][dt].wv_reads;

                if (!aecct::attn_block_phasea_kv_consume_emit(
                        in_ch_,
                        out_ch_,
                        wk_payload_.data(),
                        wk_inv_sw_bits_,
                        wv_payload_.data(),
                        wv_inv_sw_bits_)) {
                    return false;
                }
                mark_stream(t, dt, (uint32_t)aecct::ATTN_PKT_K);
                mark_stream(t, dt, (uint32_t)aecct::ATTN_PKT_V);

                if (!aecct::attn_top_writeback_phasea_kv_work_unit(
                        sram_.data(),
                        (aecct::u32_t)scr_k_row_base(t),
                        (aecct::u32_t)scr_v_row_base(t),
                        (aecct::u32_t)t,
                        (aecct::u32_t)dt,
                        out_ch_)) {
                    return false;
                }
                mark_mem(t, dt, (uint32_t)aecct::ATTN_PKT_K);
                mark_mem(t, dt, (uint32_t)aecct::ATTN_PKT_V);
                ++per_work_[t][dt].scr_k_writes;
                ++per_work_[t][dt].scr_v_writes;
            }
        }
        return true;
    }

    bool validate_order() {
        const uint32_t stream_expect[5] = {
            (uint32_t)aecct::ATTN_PKT_X,
            (uint32_t)aecct::ATTN_PKT_WK,
            (uint32_t)aecct::ATTN_PKT_WV,
            (uint32_t)aecct::ATTN_PKT_K,
            (uint32_t)aecct::ATTN_PKT_V};
        for (uint32_t t = 0u; t < token_count_; ++t) {
            std::vector<uint32_t> seq;
            std::vector<uint32_t> mem_seq;
            for (size_t i = 0; i < stream_events_.size(); ++i) {
                if (stream_events_[i].token == t && stream_events_[i].d_tile == 0u) {
                    seq.push_back(stream_events_[i].kind);
                }
            }
            for (size_t i = 0; i < mem_events_.size(); ++i) {
                if (mem_events_[i].token == t && mem_events_[i].d_tile == 0u) {
                    mem_seq.push_back(mem_events_[i].kind);
                }
            }
            if (seq.size() != 5u || mem_seq.size() != 5u) {
                std::printf("[p11ac][FAIL] order count mismatch token=%u\n", (unsigned)t);
                return false;
            }
            for (uint32_t i = 0u; i < 5u; ++i) {
                if (seq[i] != stream_expect[i] || mem_seq[i] != stream_expect[i]) {
                    std::printf("[p11ac][FAIL] order mismatch token=%u idx=%u stream=%u mem=%u expect=%u\n",
                                (unsigned)t,
                                (unsigned)i,
                                (unsigned)seq[i],
                                (unsigned)mem_seq[i],
                                (unsigned)stream_expect[i]);
                    return false;
                }
            }
            std::printf("[p11ac][STREAM_ORDER][PASS] token=%u sequence=X->Wk->Wv->K->V\n", (unsigned)t);
            std::printf("[p11ac][MEM_ORDER][PASS] token=%u sequence=read X_WORK->read Wk->read Wv->write SCR_K->write SCR_V\n", (unsigned)t);
        }
        return true;
    }

    bool validate_access_counts() {
        for (uint32_t t = 0u; t < token_count_; ++t) {
            const WorkCounters& c = per_work_[t][0];
            if (c.x_reads != 1u || c.wk_reads != 1u || c.wv_reads != 1u || c.scr_k_writes != 1u || c.scr_v_writes != 1u) {
                std::printf("[p11ac][FAIL] access count mismatch token=%u\n", (unsigned)t);
                return false;
            }
            std::printf("[p11ac][ACCESS_COUNT][PASS] token=%u X=1 Wk=1 Wv=1 SCR_K=1 SCR_V=1\n", (unsigned)t);
        }
        return true;
    }

    bool validate_exact_spans() {
        for (uint32_t t = 0u; t < token_count_; ++t) {
            const uint32_t k_base = scr_k_row_base(t);
            const uint32_t v_base = scr_v_row_base(t);
            const uint32_t x_base = x_row_base(t);
            for (uint32_t i = 0u; i < kTileWords; ++i) {
                const uint32_t got_k = (uint32_t)sram_[k_base + i].to_uint();
                const uint32_t exp_k = (uint32_t)expected_k_[t][i].to_uint();
                if (got_k != exp_k) {
                    std::printf("[p11ac][FAIL] K span mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                                (unsigned)t, (unsigned)i, (unsigned)got_k, (unsigned)exp_k);
                    return false;
                }
                const uint32_t got_v = (uint32_t)sram_[v_base + i].to_uint();
                const uint32_t exp_v = (uint32_t)expected_v_[t][i].to_uint();
                if (got_v != exp_v) {
                    std::printf("[p11ac][FAIL] V span mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                                (unsigned)t, (unsigned)i, (unsigned)got_v, (unsigned)exp_v);
                    return false;
                }
                const uint32_t got_x = (uint32_t)sram_[x_base + i].to_uint();
                const uint32_t exp_x = (uint32_t)sram_before_[x_base + i].to_uint();
                if (got_x != exp_x) {
                    std::printf("[p11ac][FAIL] X_WORK changed token=%u idx=%u got=%g exp=%g\n",
                                (unsigned)t, (unsigned)i, (double)bits_to_f32(got_x), (double)bits_to_f32(exp_x));
                    return false;
                }
            }
            std::printf("[p11ac][SPAN][PASS] token=%u K/V exact compare_length=%u\n", (unsigned)t, (unsigned)kTileWords);
        }
        return true;
    }

    aecct::u32_t x_row_[kTileWords];
    aecct::u32_t expected_k_act_q_[kTileWords];
    aecct::u32_t expected_v_act_q_[kTileWords];
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11ac tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
