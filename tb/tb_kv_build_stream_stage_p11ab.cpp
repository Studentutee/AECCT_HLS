// P00-011AB: Phase-A K/V materialization staging proof into SCR_K/SCR_V (local-only).
// Scope: stream-order, memory-access-order, X reuse, exact writeback compare, no-spurious-write checks.

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "AecctTypes.h"
#include "AecctUtil.h"
#include "blocks/TernaryLiveQkvLeafKernel.h"
#include "blocks/TernaryLiveQkvLeafKernelShapeConfig.h"
#include "blocks/TernaryLiveQkvLeafKernelTop.h"
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
static const uint32_t kExpectedTileWords = 32u;

enum StreamEventKind : uint32_t {
    EVT_STREAM_X = 0u,
    EVT_STREAM_WK = 1u,
    EVT_STREAM_WV = 2u,
    EVT_STREAM_K = 3u,
    EVT_STREAM_V = 4u
};

enum MemEventKind : uint32_t {
    EVT_MEM_READ_X_WORK = 0u,
    EVT_MEM_READ_WK = 1u,
    EVT_MEM_READ_WV = 2u,
    EVT_MEM_WRITE_SCR_K = 3u,
    EVT_MEM_WRITE_SCR_V = 4u
};

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

    WorkCounters()
        : x_reads(0u), wk_reads(0u), wv_reads(0u), scr_k_writes(0u), scr_v_writes(0u) {}
};

struct TilePacket {
    uint32_t kind;
    uint32_t token;
    uint32_t d_tile;
    std::vector<aecct::u32_t> words;
    aecct::u32_t inv_sw_bits;

    TilePacket() : kind(0u), token(0u), d_tile(0u), inv_sw_bits((aecct::u32_t)0u) {}
};

struct SramWriteRecord {
    uint32_t addr_w;
    uint32_t token;
    uint32_t d_tile;
    uint32_t kind;
};

class TileChannel {
public:
    void write(const TilePacket& p) { q_.push_back(p); }

    bool nb_read(TilePacket& out) {
        if (q_.empty()) {
            return false;
        }
        out = q_.front();
        q_.pop_front();
        return true;
    }

    bool empty() const { return q_.empty(); }

private:
    std::deque<TilePacket> q_;
};

static int fail(const char* msg) {
    std::printf("[p11ab][FAIL] %s\n", msg);
    return 1;
}

static int failf(const char* fmt, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    std::printf("[p11ab][FAIL] ");
    std::printf(fmt, (unsigned)a, (unsigned)b, (unsigned)c, (unsigned)d);
    std::printf("\n");
    return 1;
}

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

static bool validate_meta(const QuantLinearMeta& meta, uint32_t matrix_id) {
    if (meta.matrix_id != matrix_id) {
        return false;
    }
    if (meta.layout_kind != (uint32_t)QLAYOUT_TERNARY_W_OUT_IN) {
        return false;
    }
    if (meta.rows != kExpectedTileWords || meta.cols != kExpectedTileWords) {
        return false;
    }
    if (meta.num_weights != (meta.rows * meta.cols)) {
        return false;
    }
    if (meta.payload_words_2b != ternary_payload_words_2b(meta.num_weights)) {
        return false;
    }
    if (meta.last_word_valid_count == 0u || meta.last_word_valid_count > 16u) {
        return false;
    }
    if (meta.last_word_valid_count != ternary_last_word_valid_count(meta.num_weights)) {
        return false;
    }
    return true;
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

static bool write_payload_and_inv_to_sram(
    aecct::u32_t* sram,
    uint32_t param_base_word,
    uint32_t matrix_id,
    const QuantLinearMeta& meta,
    const std::vector<aecct::u32_t>& payload_words,
    uint32_t& out_inv_sw_bits) {
    const ParamMeta payload_meta = kParamMeta[meta.weight_param_id];
    const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];
    if (payload_meta.len_w < meta.payload_words_2b || inv_meta.len_w == 0u) {
        return false;
    }
    for (uint32_t i = 0u; i < meta.payload_words_2b; ++i) {
        sram[param_base_word + payload_meta.offset_w + i] = payload_words[i];
    }

    double inv_sw_scale = 0.0;
    if (!matrix_inv_sw(matrix_id, inv_sw_scale)) {
        return false;
    }
    out_inv_sw_bits = (uint32_t)aecct::fp32_bits_from_double(1.0 / inv_sw_scale).to_uint();
    sram[param_base_word + inv_meta.offset_w] = (aecct::u32_t)out_inv_sw_bits;
    return true;
}

static void fill_x_work_fixture(
    std::vector<aecct::u32_t>& sram,
    uint32_t x_base_word,
    uint32_t tile_words) {
    for (uint32_t t = 0u; t < kTokenCount; ++t) {
        const uint32_t row_base = x_base_word + t * tile_words;
        for (uint32_t i = 0u; i < tile_words; ++i) {
            const int32_t v = (int32_t)((t + 1u) * 17u + (i * 3u)) - 29;
            const float f = ((float)v) * 0.03125f;
            sram[row_base + i] = (aecct::u32_t)f32_to_bits(f);
        }
    }
}

static void collect_work_unit_events(
    const std::vector<EventTag>& all,
    uint32_t token,
    uint32_t d_tile,
    std::vector<uint32_t>& out_kinds) {
    out_kinds.clear();
    for (size_t i = 0; i < all.size(); ++i) {
        if (all[i].token == token && all[i].d_tile == d_tile) {
            out_kinds.push_back(all[i].kind);
        }
    }
}

static bool compare_exact_word(
    const char* label,
    uint32_t got_bits,
    uint32_t expect_bits,
    uint32_t idx) {
    if (got_bits != expect_bits) {
        std::printf(
            "[p11ab][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
            label,
            (unsigned)idx,
            (unsigned)got_bits,
            (unsigned)expect_bits,
            (double)bits_to_f32(got_bits),
            (double)bits_to_f32(expect_bits));
        return false;
    }
    return true;
}

class TbP11ab {
public:
    int run_all() {
        if (!derive_and_assert_tile_words()) {
            return fail("tile-width authority assertion failed");
        }

        if (!prepare_meta_and_payload()) {
            return fail("WK/WV payload authority preparation failed");
        }

        if (!init_sram()) {
            return fail("SRAM fixture initialization failed");
        }
        if (!compute_expected_rows()) {
            return fail("expected path compute failed");
        }

        if (!run_stream_staging()) {
            return fail("stream staging flow failed");
        }

        if (!validate_stream_order()) {
            return 1;
        }
        if (!validate_mem_order()) {
            return 1;
        }
        if (!validate_access_counts()) {
            return 1;
        }
        if (!validate_x_reuse()) {
            return 1;
        }
        if (!validate_span_compares()) {
            return 1;
        }
        if (!validate_x_work_unchanged()) {
            return 1;
        }
        if (!validate_no_spurious_writes()) {
            return 1;
        }

        std::printf("PASS: tb_kv_build_stream_stage_p11ab\n");
        return 0;
    }

private:
    uint32_t tile_words_;
    uint32_t x_base_word_;
    uint32_t scr_k_base_word_;
    uint32_t scr_v_base_word_;
    uint32_t param_base_word_;
    uint32_t tmp_out_base_word_;
    uint32_t tmp_out_act_base_word_;

    QuantLinearMeta wk_meta_;
    QuantLinearMeta wv_meta_;
    std::vector<aecct::u32_t> wk_payload_;
    std::vector<aecct::u32_t> wv_payload_;
    uint32_t wk_inv_sw_bits_;
    uint32_t wv_inv_sw_bits_;

    std::vector<aecct::u32_t> sram_seed_;
    std::vector<aecct::u32_t> sram_dut_;
    std::vector<aecct::u32_t> x_work_before_;

    std::vector< std::vector<aecct::u32_t> > expected_k_;
    std::vector< std::vector<aecct::u32_t> > expected_v_;

    TileChannel in_ch_;
    TileChannel out_ch_;
    WorkCounters per_work_[kTokenCount][kDTileCount];
    std::vector<EventTag> stream_events_;
    std::vector<EventTag> mem_events_;
    std::vector<SramWriteRecord> runtime_writes_;

    uint32_t total_x_reads_;
    uint32_t total_wk_reads_;
    uint32_t total_wv_reads_;
    uint32_t total_scr_k_writes_;
    uint32_t total_scr_v_writes_;

public:
    TbP11ab()
        : tile_words_(0u),
          x_base_word_(0u),
          scr_k_base_word_(0u),
          scr_v_base_word_(0u),
          param_base_word_(0u),
          tmp_out_base_word_(0u),
          tmp_out_act_base_word_(0u),
          wk_inv_sw_bits_(0u),
          wv_inv_sw_bits_(0u),
          total_x_reads_(0u),
          total_wk_reads_(0u),
          total_wv_reads_(0u),
          total_scr_k_writes_(0u),
          total_scr_v_writes_(0u) {}

    bool derive_and_assert_tile_words() {
        // Tile width is derived from accepted compile-time shape SSOT / existing top constants.
        if (aecct::kQkvCtSupportedL0WkCols != aecct::kQkvCtSupportedL0WvCols) {
            return false;
        }
        tile_words_ = aecct::kQkvCtSupportedL0WkCols;
        if (tile_words_ != kExpectedTileWords) {
            std::printf("[p11ab][FAIL] fixture tile words expected=%u got=%u\n",
                        (unsigned)kExpectedTileWords,
                        (unsigned)tile_words_);
            return false;
        }
        std::printf("[p11ab][FIXTURE][PASS] derived d_tile words=%u from accepted compile-time shape SSOT\n",
                    (unsigned)tile_words_);

        x_base_word_ = (uint32_t)sram_map::BASE_X_WORK_W;
        scr_k_base_word_ = (uint32_t)sram_map::BASE_SCR_K_W;
        scr_v_base_word_ = (uint32_t)sram_map::BASE_SCR_V_W;
        param_base_word_ = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
        tmp_out_base_word_ = (uint32_t)sram_map::BASE_SCR_FINAL_SCALAR_W;
        tmp_out_act_base_word_ = tmp_out_base_word_ + tile_words_;
        return true;
    }

    bool prepare_meta_and_payload() {
        wk_meta_ = kQuantLinearMeta[(uint32_t)QLM_L0_WK];
        wv_meta_ = kQuantLinearMeta[(uint32_t)QLM_L0_WV];
        if (!validate_meta(wk_meta_, (uint32_t)QLM_L0_WK)) {
            return false;
        }
        if (!validate_meta(wv_meta_, (uint32_t)QLM_L0_WV)) {
            return false;
        }

        if (!build_payload_for_matrix((uint32_t)QLM_L0_WK, wk_meta_, wk_payload_)) {
            return false;
        }
        if (!build_payload_for_matrix((uint32_t)QLM_L0_WV, wv_meta_, wv_payload_)) {
            return false;
        }
        return true;
    }

    bool init_sram() {
        sram_seed_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        sram_dut_.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);

        fill_x_work_fixture(sram_seed_, x_base_word_, tile_words_);

        bool ok_wk = write_payload_and_inv_to_sram(
            sram_seed_.data(),
            param_base_word_,
            (uint32_t)QLM_L0_WK,
            wk_meta_,
            wk_payload_,
            wk_inv_sw_bits_);
        bool ok_wv = write_payload_and_inv_to_sram(
            sram_seed_.data(),
            param_base_word_,
            (uint32_t)QLM_L0_WV,
            wv_meta_,
            wv_payload_,
            wv_inv_sw_bits_);
        if (!ok_wk || !ok_wv) {
            std::printf("[p11ab][FAIL] payload write-to-SRAM failed\n");
            return false;
        }

        sram_dut_ = sram_seed_;

        x_work_before_.assign(kTokenCount * tile_words_, (aecct::u32_t)0u);
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t x_row_base = x_base_word_ + t * tile_words_;
            for (uint32_t i = 0u; i < tile_words_; ++i) {
                x_work_before_[t * tile_words_ + i] = sram_dut_[x_row_base + i];
            }
        }
        return true;
    }

    bool compute_expected_rows() {
        expected_k_.assign(kTokenCount, std::vector<aecct::u32_t>(tile_words_, (aecct::u32_t)0u));
        expected_v_.assign(kTokenCount, std::vector<aecct::u32_t>(tile_words_, (aecct::u32_t)0u));

        std::vector<aecct::u32_t> sram_ref = sram_seed_;
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t x_row_base = x_base_word_ + t * tile_words_;
            aecct::u32_t wk_inv = (aecct::u32_t)0u;
            if (!aecct::ternary_live_l0_wk_materialize_row_kernel(
                    sram_ref.data(),
                    (aecct::u32_t)param_base_word_,
                    (aecct::u32_t)x_row_base,
                    (aecct::u32_t)tmp_out_base_word_,
                    (aecct::u32_t)tmp_out_act_base_word_,
                    wk_inv)) {
                return false;
            }
            for (uint32_t i = 0u; i < tile_words_; ++i) {
                expected_k_[t][i] = sram_ref[tmp_out_base_word_ + i];
            }

            aecct::u32_t wv_inv = (aecct::u32_t)0u;
            if (!aecct::ternary_live_l0_wv_materialize_row_kernel(
                    sram_ref.data(),
                    (aecct::u32_t)param_base_word_,
                    (aecct::u32_t)x_row_base,
                    (aecct::u32_t)tmp_out_base_word_,
                    (aecct::u32_t)tmp_out_act_base_word_,
                    wv_inv)) {
                return false;
            }
            for (uint32_t i = 0u; i < tile_words_; ++i) {
                expected_v_[t][i] = sram_ref[tmp_out_base_word_ + i];
            }
        }
        return true;
    }

    void mark_stream_event(uint32_t token, uint32_t d_tile, uint32_t kind) {
        EventTag e;
        e.token = token;
        e.d_tile = d_tile;
        e.kind = kind;
        stream_events_.push_back(e);
    }

    void mark_mem_event(uint32_t token, uint32_t d_tile, uint32_t kind) {
        EventTag e;
        e.token = token;
        e.d_tile = d_tile;
        e.kind = kind;
        mem_events_.push_back(e);
    }

    uint32_t x_row_base(uint32_t token, uint32_t d_tile) const {
        (void)d_tile;
        return x_base_word_ + token * tile_words_;
    }

    uint32_t scr_k_row_base(uint32_t token, uint32_t d_tile) const {
        (void)d_tile;
        return scr_k_base_word_ + token * tile_words_;
    }

    uint32_t scr_v_row_base(uint32_t token, uint32_t d_tile) const {
        (void)d_tile;
        return scr_v_base_word_ + token * tile_words_;
    }

    bool top_issue_x(uint32_t token, uint32_t d_tile) {
        TilePacket p;
        p.kind = EVT_STREAM_X;
        p.token = token;
        p.d_tile = d_tile;
        p.words.assign(tile_words_, (aecct::u32_t)0u);

        const uint32_t base = x_row_base(token, d_tile);
        for (uint32_t i = 0u; i < tile_words_; ++i) {
            p.words[i] = sram_dut_[base + i];
        }

        in_ch_.write(p);
        mark_stream_event(token, d_tile, EVT_STREAM_X);
        mark_mem_event(token, d_tile, EVT_MEM_READ_X_WORK);
        ++per_work_[token][d_tile].x_reads;
        ++total_x_reads_;
        return true;
    }

    bool top_issue_weight_tile(uint32_t token, uint32_t d_tile, uint32_t stream_kind) {
        const QuantLinearMeta& meta =
            (stream_kind == EVT_STREAM_WK) ? wk_meta_ : wv_meta_;
        const ParamMeta payload_meta = kParamMeta[meta.weight_param_id];
        const ParamMeta inv_meta = kParamMeta[meta.inv_sw_param_id];

        TilePacket p;
        p.kind = stream_kind;
        p.token = token;
        p.d_tile = d_tile;
        p.words.assign(meta.payload_words_2b, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < meta.payload_words_2b; ++i) {
            p.words[i] = sram_dut_[param_base_word_ + payload_meta.offset_w + i];
        }
        p.inv_sw_bits = sram_dut_[param_base_word_ + inv_meta.offset_w];

        in_ch_.write(p);
        mark_stream_event(token, d_tile, stream_kind);
        if (stream_kind == EVT_STREAM_WK) {
            mark_mem_event(token, d_tile, EVT_MEM_READ_WK);
            ++per_work_[token][d_tile].wk_reads;
            ++total_wk_reads_;
        } else {
            mark_mem_event(token, d_tile, EVT_MEM_READ_WV);
            ++per_work_[token][d_tile].wv_reads;
            ++total_wv_reads_;
        }
        return true;
    }

    bool block_consume_and_emit(uint32_t token, uint32_t d_tile) {
        TilePacket x_pkt;
        TilePacket wk_pkt;
        TilePacket wv_pkt;
        if (!in_ch_.nb_read(x_pkt) || !in_ch_.nb_read(wk_pkt) || !in_ch_.nb_read(wv_pkt)) {
            return false;
        }
        if (x_pkt.kind != EVT_STREAM_X || wk_pkt.kind != EVT_STREAM_WK || wv_pkt.kind != EVT_STREAM_WV) {
            return false;
        }
        if (x_pkt.token != token || wk_pkt.token != token || wv_pkt.token != token) {
            return false;
        }
        if (x_pkt.d_tile != d_tile || wk_pkt.d_tile != d_tile || wv_pkt.d_tile != d_tile) {
            return false;
        }
        if (x_pkt.words.size() != tile_words_) {
            return false;
        }
        if (wk_pkt.words.size() != wk_meta_.payload_words_2b || wv_pkt.words.size() != wv_meta_.payload_words_2b) {
            return false;
        }

        // Tile-local X retention for WK then WV.
        aecct::u32_t x_row_wk[aecct::kTernaryLiveL0WkCols];
        aecct::u32_t wk_payload[aecct::kTernaryLiveL0WkPayloadWords];
        aecct::u32_t k_out[aecct::kTernaryLiveL0WkRows];
        aecct::u32_t k_out_act_q[aecct::kTernaryLiveL0WkRows];
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkCols; ++i) {
            x_row_wk[i] = x_pkt.words[i];
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WkPayloadWords; ++i) {
            wk_payload[i] = wk_pkt.words[i];
        }
        aecct::u32_t wk_inv = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WkRowTop wk_top;
        if (!wk_top.run(x_row_wk,
                        wk_payload,
                        wk_pkt.inv_sw_bits,
                        k_out,
                        k_out_act_q,
                        wk_inv)) {
            return false;
        }

        TilePacket k_pkt;
        k_pkt.kind = EVT_STREAM_K;
        k_pkt.token = token;
        k_pkt.d_tile = d_tile;
        k_pkt.words.assign(tile_words_, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < tile_words_; ++i) {
            k_pkt.words[i] = k_out[i];
        }
        out_ch_.write(k_pkt);
        mark_stream_event(token, d_tile, EVT_STREAM_K);

        aecct::u32_t x_row_wv[aecct::kTernaryLiveL0WvCols];
        aecct::u32_t wv_payload[aecct::kTernaryLiveL0WvPayloadWords];
        aecct::u32_t v_out[aecct::kTernaryLiveL0WvRows];
        aecct::u32_t v_out_act_q[aecct::kTernaryLiveL0WvRows];
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvCols; ++i) {
            // Reuse the retained X tile directly.
            x_row_wv[i] = x_pkt.words[i];
        }
        for (uint32_t i = 0u; i < aecct::kTernaryLiveL0WvPayloadWords; ++i) {
            wv_payload[i] = wv_pkt.words[i];
        }
        aecct::u32_t wv_inv = (aecct::u32_t)0u;
        aecct::TernaryLiveL0WvRowTop wv_top;
        if (!wv_top.run(x_row_wv,
                        wv_payload,
                        wv_pkt.inv_sw_bits,
                        v_out,
                        v_out_act_q,
                        wv_inv)) {
            return false;
        }

        TilePacket v_pkt;
        v_pkt.kind = EVT_STREAM_V;
        v_pkt.token = token;
        v_pkt.d_tile = d_tile;
        v_pkt.words.assign(tile_words_, (aecct::u32_t)0u);
        for (uint32_t i = 0u; i < tile_words_; ++i) {
            v_pkt.words[i] = v_out[i];
        }
        out_ch_.write(v_pkt);
        mark_stream_event(token, d_tile, EVT_STREAM_V);

        return true;
    }

    bool top_writeback(uint32_t token, uint32_t d_tile) {
        TilePacket k_pkt;
        TilePacket v_pkt;
        if (!out_ch_.nb_read(k_pkt) || !out_ch_.nb_read(v_pkt)) {
            return false;
        }
        if (k_pkt.kind != EVT_STREAM_K || v_pkt.kind != EVT_STREAM_V) {
            return false;
        }
        if (k_pkt.token != token || v_pkt.token != token || k_pkt.d_tile != d_tile || v_pkt.d_tile != d_tile) {
            return false;
        }

        const uint32_t k_base = scr_k_row_base(token, d_tile);
        for (uint32_t i = 0u; i < tile_words_; ++i) {
            sram_dut_[k_base + i] = k_pkt.words[i];
            SramWriteRecord wr;
            wr.addr_w = k_base + i;
            wr.token = token;
            wr.d_tile = d_tile;
            wr.kind = EVT_MEM_WRITE_SCR_K;
            runtime_writes_.push_back(wr);
        }
        mark_mem_event(token, d_tile, EVT_MEM_WRITE_SCR_K);
        ++per_work_[token][d_tile].scr_k_writes;
        ++total_scr_k_writes_;

        const uint32_t v_base = scr_v_row_base(token, d_tile);
        for (uint32_t i = 0u; i < tile_words_; ++i) {
            sram_dut_[v_base + i] = v_pkt.words[i];
            SramWriteRecord wr;
            wr.addr_w = v_base + i;
            wr.token = token;
            wr.d_tile = d_tile;
            wr.kind = EVT_MEM_WRITE_SCR_V;
            runtime_writes_.push_back(wr);
        }
        mark_mem_event(token, d_tile, EVT_MEM_WRITE_SCR_V);
        ++per_work_[token][d_tile].scr_v_writes;
        ++total_scr_v_writes_;

        return true;
    }

    bool run_stream_staging() {
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            for (uint32_t dt = 0u; dt < kDTileCount; ++dt) {
                if (!top_issue_x(t, dt)) {
                    return false;
                }
                if (!top_issue_weight_tile(t, dt, EVT_STREAM_WK)) {
                    return false;
                }
                if (!top_issue_weight_tile(t, dt, EVT_STREAM_WV)) {
                    return false;
                }
                if (!block_consume_and_emit(t, dt)) {
                    return false;
                }
                if (!top_writeback(t, dt)) {
                    return false;
                }
            }
        }
        return in_ch_.empty() && out_ch_.empty();
    }

    bool validate_stream_order() {
        const uint32_t expect[5] = {EVT_STREAM_X, EVT_STREAM_WK, EVT_STREAM_WV, EVT_STREAM_K, EVT_STREAM_V};
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            for (uint32_t dt = 0u; dt < kDTileCount; ++dt) {
                std::vector<uint32_t> seq;
                collect_work_unit_events(stream_events_, t, dt, seq);
                if (seq.size() != 5u) {
                    failf("stream-order event count mismatch token=%u d_tile=%u got=%u expect=%u",
                          t, dt, (uint32_t)seq.size(), 5u);
                    return false;
                }
                for (uint32_t i = 0u; i < 5u; ++i) {
                    if (seq[i] != expect[i]) {
                        failf("stream-order mismatch token=%u d_tile=%u idx=%u got_kind=%u",
                              t, dt, i, seq[i]);
                        return false;
                    }
                }
                std::printf("[p11ab][STREAM_ORDER][PASS] token=%u d_tile=%u sequence=X->Wk->Wv->K->V\n",
                            (unsigned)t,
                            (unsigned)dt);
            }
        }
        return true;
    }

    bool validate_mem_order() {
        const uint32_t expect[5] = {
            EVT_MEM_READ_X_WORK, EVT_MEM_READ_WK, EVT_MEM_READ_WV, EVT_MEM_WRITE_SCR_K, EVT_MEM_WRITE_SCR_V};
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            for (uint32_t dt = 0u; dt < kDTileCount; ++dt) {
                std::vector<uint32_t> seq;
                collect_work_unit_events(mem_events_, t, dt, seq);
                if (seq.size() != 5u) {
                    failf("mem-order event count mismatch token=%u d_tile=%u got=%u expect=%u",
                          t, dt, (uint32_t)seq.size(), 5u);
                    return false;
                }
                for (uint32_t i = 0u; i < 5u; ++i) {
                    if (seq[i] != expect[i]) {
                        failf("mem-order mismatch token=%u d_tile=%u idx=%u got_kind=%u",
                              t, dt, i, seq[i]);
                        return false;
                    }
                }
                std::printf(
                    "[p11ab][MEM_ORDER][PASS] token=%u d_tile=%u sequence=read X_WORK->read W_REGION(Wk)->read W_REGION(Wv)->write SCR_K->write SCR_V\n",
                    (unsigned)t,
                    (unsigned)dt);
            }
        }
        return true;
    }

    bool validate_access_counts() {
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            for (uint32_t dt = 0u; dt < kDTileCount; ++dt) {
                const WorkCounters& c = per_work_[t][dt];
                if (c.x_reads != 1u || c.wk_reads != 1u || c.wv_reads != 1u ||
                    c.scr_k_writes != 1u || c.scr_v_writes != 1u) {
                    std::printf(
                        "[p11ab][FAIL] access-count mismatch token=%u d_tile=%u X=%u Wk=%u Wv=%u SCR_K=%u SCR_V=%u\n",
                        (unsigned)t,
                        (unsigned)dt,
                        (unsigned)c.x_reads,
                        (unsigned)c.wk_reads,
                        (unsigned)c.wv_reads,
                        (unsigned)c.scr_k_writes,
                        (unsigned)c.scr_v_writes);
                    return false;
                }
                std::printf(
                    "[p11ab][ACCESS_COUNT][PASS] token=%u d_tile=%u X=1 Wk=1 Wv=1 SCR_K=1 SCR_V=1\n",
                    (unsigned)t,
                    (unsigned)dt);
            }
        }

        std::printf(
            "[p11ab][ACCESS_TOTAL] x_reads=%u wk_reads=%u wv_reads=%u scr_k_writes=%u scr_v_writes=%u\n",
            (unsigned)total_x_reads_,
            (unsigned)total_wk_reads_,
            (unsigned)total_wv_reads_,
            (unsigned)total_scr_k_writes_,
            (unsigned)total_scr_v_writes_);
        return true;
    }

    bool validate_x_reuse() {
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            for (uint32_t dt = 0u; dt < kDTileCount; ++dt) {
                const uint32_t duplicate_x_reads = (per_work_[t][dt].x_reads > 0u)
                                                       ? (per_work_[t][dt].x_reads - 1u)
                                                       : 0u;
                if (duplicate_x_reads != 0u) {
                    std::printf("[p11ab][FAIL] duplicate X_WORK read token=%u d_tile=%u duplicate=%u\n",
                                (unsigned)t,
                                (unsigned)dt,
                                (unsigned)duplicate_x_reads);
                    return false;
                }
                std::printf(
                    "[p11ab][X_REUSE][PASS] token=%u d_tile=%u same X tile retained/reused for K and V; duplicate X_WORK read=0\n",
                    (unsigned)t,
                    (unsigned)dt);
            }
        }
        return true;
    }

    bool compare_one_span(
        uint32_t token,
        uint32_t d_tile,
        const char* kind,
        uint32_t target_offset,
        const std::vector<aecct::u32_t>& expected_words) {
        const uint32_t len = tile_words_;
        for (uint32_t i = 0u; i < len; ++i) {
            const uint32_t got = (uint32_t)sram_dut_[target_offset + i].to_uint();
            const uint32_t exp = (uint32_t)expected_words[i].to_uint();
            if (got != exp) {
                std::printf(
                    "[p11ab][SPAN][FAIL] token_index=%u d_tile_index=%u kind=%s target_offset=%u compare_length=%u result=FAIL first_mismatch_index=%u expected=0x%08X actual=0x%08X\n",
                    (unsigned)token,
                    (unsigned)d_tile,
                    kind,
                    (unsigned)target_offset,
                    (unsigned)len,
                    (unsigned)i,
                    (unsigned)exp,
                    (unsigned)got);
                return false;
            }
        }
        std::printf(
            "[p11ab][SPAN][PASS] token_index=%u d_tile_index=%u kind=%s target_offset=%u compare_length=%u result=PASS\n",
            (unsigned)token,
            (unsigned)d_tile,
            kind,
            (unsigned)target_offset,
            (unsigned)len);
        return true;
    }

    bool validate_span_compares() {
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            for (uint32_t dt = 0u; dt < kDTileCount; ++dt) {
                if (!compare_one_span(
                        t,
                        dt,
                        "K",
                        scr_k_row_base(t, dt),
                        expected_k_[t])) {
                    return false;
                }
                if (!compare_one_span(
                        t,
                        dt,
                        "V",
                        scr_v_row_base(t, dt),
                        expected_v_[t])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool validate_x_work_unchanged() {
        for (uint32_t t = 0u; t < kTokenCount; ++t) {
            const uint32_t base = x_row_base(t, 0u);
            for (uint32_t i = 0u; i < tile_words_; ++i) {
                const uint32_t got = (uint32_t)sram_dut_[base + i].to_uint();
                const uint32_t exp = (uint32_t)x_work_before_[t * tile_words_ + i].to_uint();
                if (!compare_exact_word("X_WORK unchanged", got, exp, i)) {
                    std::printf("[p11ab][FAIL] X_WORK changed token=%u addr=%u\n",
                                (unsigned)t,
                                (unsigned)(base + i));
                    return false;
                }
            }
        }
        std::printf("[p11ab][X_WORK][PASS] source span unchanged across Phase-A staging work-units\n");
        return true;
    }

    bool is_in_range(uint32_t addr_w, uint32_t base_w, uint32_t size_w) const {
        return (addr_w >= base_w) && (addr_w < (base_w + size_w));
    }

    bool validate_no_spurious_writes() {
        const uint32_t expected_total_words = kTokenCount * kDTileCount * tile_words_ * 2u;
        if (runtime_writes_.size() != expected_total_words) {
            std::printf("[p11ab][FAIL] runtime write count mismatch got=%u expect=%u\n",
                        (unsigned)runtime_writes_.size(),
                        (unsigned)expected_total_words);
            return false;
        }

        for (size_t i = 0; i < runtime_writes_.size(); ++i) {
            const SramWriteRecord& wr = runtime_writes_[i];
            if (is_in_range(wr.addr_w, (uint32_t)sram_map::BASE_X_WORK_W, (uint32_t)sram_map::SIZE_X_WORK_W)) {
                std::printf("[p11ab][FAIL] forbidden write to X_WORK addr=%u\n", (unsigned)wr.addr_w);
                return false;
            }
            if (is_in_range(wr.addr_w, (uint32_t)sram_map::W_REGION_BASE, (uint32_t)sram_map::W_REGION_WORDS)) {
                std::printf("[p11ab][FAIL] forbidden write to W_REGION addr=%u\n", (unsigned)wr.addr_w);
                return false;
            }

            const uint32_t k_base = scr_k_row_base(wr.token, wr.d_tile);
            const uint32_t v_base = scr_v_row_base(wr.token, wr.d_tile);
            const bool in_k = is_in_range(wr.addr_w, k_base, tile_words_);
            const bool in_v = is_in_range(wr.addr_w, v_base, tile_words_);
            if (!in_k && !in_v) {
                std::printf("[p11ab][FAIL] write outside allowed SCR_K/SCR_V spans addr=%u token=%u d_tile=%u\n",
                            (unsigned)wr.addr_w,
                            (unsigned)wr.token,
                            (unsigned)wr.d_tile);
                return false;
            }
            if (wr.kind == EVT_MEM_WRITE_SCR_K && !in_k) {
                std::printf("[p11ab][FAIL] SCR_K-tagged write landed outside SCR_K span addr=%u\n",
                            (unsigned)wr.addr_w);
                return false;
            }
            if (wr.kind == EVT_MEM_WRITE_SCR_V && !in_v) {
                std::printf("[p11ab][FAIL] SCR_V-tagged write landed outside SCR_V span addr=%u\n",
                            (unsigned)wr.addr_w);
                return false;
            }
        }

        std::printf(
            "[p11ab][WRITE_GUARD][PASS] no writes to X_WORK/W_REGION/unrelated scratch; only SCR_K[token,d_tile] and SCR_V[token,d_tile] writes observed\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11ab tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
