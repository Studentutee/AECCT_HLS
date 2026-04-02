// P00-011AJ: Top-managed SRAM full-flow provenance + final compare hardening (local-only).
// Scope:
// - Provenance checks for AC/AD/AE/AF stage spans.
// - Downstream bridge checks (Q/K->score, score->output, attn_out->final_x).
// - Full-loop vs staged-flow key-span exact compare with hardened final_x compare.

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdint>
#include <vector>

#include "tb_p11aeaf_common.h"

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

class TbP11ajTopManagedSramProvenance {
public:
    int run_all() {
        if (!init_state()) {
            return 1;
        }
        if (!run_staged_provenance_chain()) {
            return 1;
        }
        if (!run_staged_downstream_finalize()) {
            return 1;
        }
        if (!run_direct_attnout_to_finalx_bridge_probe()) {
            return 1;
        }
        if (!run_full_loop_mainline()) {
            return 1;
        }
        if (!run_full_loop_mixed_layer_mainline()) {
            return 1;
        }
        if (!validate_full_flow_key_span_compare()) {
            return 1;
        }
        if (!validate_final_x_hardened_compare()) {
            return 1;
        }
        std::printf("PASS: tb_attnout_finalx_bridge_p11ak\n");
        std::printf("PASS: tb_top_managed_sram_provenance_p11aj\n");
        return 0;
    }

private:
    std::vector<aecct::u32_t> sram_stage_;
    std::vector<aecct::u32_t> sram_full_;
    std::vector<aecct::u32_t> sram_stage_before_downstream_;
    p11aeaf_tb::QkvPayloadSet payloads_;
    aecct::CfgRegs cfg_;
    aecct::LayerScratch sc_;
    aecct::TopRegs regs_full_;
    uint32_t param_base_;
    uint32_t token_count_;
    uint32_t d_model_;
    uint32_t n_heads_;
    uint32_t d_head_;
    uint32_t staged_final_x_base_;
    bool ae_any_score_change_;
    bool af_any_pre_change_;
    bool af_any_post_change_;
    bool af_any_out_change_;

    static uint32_t f32_to_bits(float f) {
        union {
            float f;
            uint32_t u;
        } cvt;
        cvt.f = f;
        return cvt.u;
    }

    static aecct::u32_t perturb_fp32_bits(aecct::u32_t in_bits) {
        const uint32_t raw = (uint32_t)in_bits.to_uint();
        return (aecct::u32_t)(raw ^ 0x00400000u);
    }

    static aecct::u32_t force_delta_bits(aecct::u32_t in_bits) {
        if ((uint32_t)in_bits.to_uint() != 0u) {
            return (aecct::u32_t)0u;
        }
        return (aecct::u32_t)0x3F800000u;
    }

    static bool is_nonfinite_bits(uint32_t bits) {
        return ((bits & 0x7F800000u) == 0x7F800000u);
    }

    void apply_bridge_probe_norm_params(std::vector<aecct::u32_t>& sram_vec) const {
        const uint32_t one_bits = 0x3F800000u;
        const uint32_t zero_bits = 0x00000000u;
        const uint32_t layer_norm_w_base = param_base_ + kParamMeta[43u].offset_w;
        const uint32_t layer_norm_b_base = param_base_ + kParamMeta[7u].offset_w;
        const uint32_t end_norm_w_base = param_base_ + kParamMeta[64u].offset_w;
        const uint32_t end_norm_b_base = param_base_ + kParamMeta[16u].offset_w;
        for (uint32_t c = 0u; c < d_model_; ++c) {
            sram_vec[layer_norm_w_base + c] = (aecct::u32_t)one_bits;
            sram_vec[layer_norm_b_base + c] = (aecct::u32_t)zero_bits;
            sram_vec[end_norm_w_base + c] = (aecct::u32_t)one_bits;
            sram_vec[end_norm_b_base + c] = (aecct::u32_t)zero_bits;
        }
    }

    static void init_full_x_rows(std::vector<aecct::u32_t>& sram) {
        const uint32_t token_count = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        const uint32_t d_model = (uint32_t)aecct::ATTN_D_MODEL;
        const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        for (uint32_t t = 0u; t < token_count; ++t) {
            const uint32_t row_base = x_base + t * d_model;
            for (uint32_t i = 0u; i < d_model; ++i) {
                const int32_t v = (int32_t)((t + 3u) * 17u + (i + 5u) * 11u) - 211;
                const float f = ((float)v) * 0.015625f;
                sram[row_base + i] = (aecct::u32_t)f32_to_bits(f);
            }
        }
    }

    static void mark_span(std::vector<uint8_t>& allowed, uint32_t base, uint32_t words) {
        for (uint32_t i = 0u; i < words; ++i) {
            allowed[base + i] = 1u;
        }
    }

    static uint32_t count_span_changes(
        const std::vector<aecct::u32_t>& before,
        const std::vector<aecct::u32_t>& after,
        uint32_t base,
        uint32_t words
    ) {
        uint32_t changed = 0u;
        for (uint32_t i = 0u; i < words; ++i) {
            if ((uint32_t)before[base + i].to_uint() != (uint32_t)after[base + i].to_uint()) {
                ++changed;
            }
        }
        return changed;
    }

    static bool span_unchanged(
        const std::vector<aecct::u32_t>& before,
        const std::vector<aecct::u32_t>& after,
        uint32_t base,
        uint32_t words,
        const char* label
    ) {
        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t bv = (uint32_t)before[base + i].to_uint();
            const uint32_t av = (uint32_t)after[base + i].to_uint();
            if (bv != av) {
                std::printf("[p11aj][FAIL] %s changed unexpectedly addr=%u offs=%u before=0x%08X after=0x%08X\n",
                    label, (unsigned)(base + i), (unsigned)i, (unsigned)bv, (unsigned)av);
                return false;
            }
        }
        return true;
    }

    static bool check_no_spurious_changes(
        const std::vector<aecct::u32_t>& before,
        const std::vector<aecct::u32_t>& after,
        const std::vector<uint8_t>& allowed,
        const char* stage_label
    ) {
        const uint32_t words = (uint32_t)before.size();
        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t bv = (uint32_t)before[i].to_uint();
            const uint32_t av = (uint32_t)after[i].to_uint();
            if (bv != av && allowed[i] == 0u) {
                std::printf("[p11aj][FAIL] %s spurious write addr=%u before=0x%08X after=0x%08X\n",
                    stage_label, (unsigned)i, (unsigned)bv, (unsigned)av);
                return false;
            }
        }
        return true;
    }

    static bool compare_span_exact(
        const std::vector<aecct::u32_t>& lhs,
        const std::vector<aecct::u32_t>& rhs,
        uint32_t base,
        uint32_t words,
        const char* label
    ) {
        for (uint32_t i = 0u; i < words; ++i) {
            const uint32_t lv = (uint32_t)lhs[base + i].to_uint();
            const uint32_t rv = (uint32_t)rhs[base + i].to_uint();
            if (lv != rv) {
                std::printf("[p11aj][FAIL] %s mismatch addr=%u offs=%u lhs=0x%08X rhs=0x%08X\n",
                    label, (unsigned)(base + i), (unsigned)i, (unsigned)lv, (unsigned)rv);
                return false;
            }
        }
        return true;
    }

    static uint32_t count_span_diffs(
        const std::vector<aecct::u32_t>& lhs,
        const std::vector<aecct::u32_t>& rhs,
        uint32_t base,
        uint32_t words
    ) {
        uint32_t diffs = 0u;
        for (uint32_t i = 0u; i < words; ++i) {
            if ((uint32_t)lhs[base + i].to_uint() != (uint32_t)rhs[base + i].to_uint()) {
                ++diffs;
            }
        }
        return diffs;
    }

    void seed_full_loop_sram(std::vector<aecct::u32_t>& sram_vec) const {
        sram_vec.assign((uint32_t)sram_map::SRAM_WORDS_TOTAL, (aecct::u32_t)0u);
        init_full_x_rows(sram_vec);
        p11aeaf_tb::load_qkv_payload_set_to_sram(sram_vec, payloads_, param_base_);
    }

    void init_top_regs_for_layers(aecct::TopRegs& regs, uint32_t n_layers) const {
        regs.clear();
        regs.w_base_set = true;
        regs.w_base_word = (aecct::u32_t)param_base_;
        regs.cfg_d_model = cfg_.d_model;
        regs.cfg_n_heads = cfg_.n_heads;
        regs.cfg_d_ffn = cfg_.d_ffn;
        regs.cfg_n_layers = (aecct::u32_t)n_layers;
        regs.cfg_ready = true;
    }

    bool check_lid0_marker_contract(
        const aecct::TopRegs& regs,
        const char* case_tag,
        bool emit_legacy_lines
    ) const {
        const bool ad_mainline = regs.p11ad_mainline_q_path_taken;
        const bool ac_mainline = regs.p11ac_mainline_path_taken;
        const bool ae_mainline = regs.p11ae_mainline_score_path_taken;
        const bool af_mainline = regs.p11af_mainline_softmax_output_path_taken;
        const bool ad_fallback = regs.p11ad_q_fallback_taken;
        const bool ac_fallback = regs.p11ac_fallback_taken;
        const bool ae_fallback = regs.p11ae_score_fallback_taken;
        const bool af_fallback = regs.p11af_softmax_output_fallback_taken;

        std::printf(
            "CASE_%s_LID0_ATTN_MAINLINE_FLAGS p11ad_mainline_q_path_taken=%u p11ac_mainline_path_taken=%u p11ae_mainline_score_path_taken=%u p11af_mainline_softmax_output_path_taken=%u\n",
            case_tag,
            ad_mainline ? 1u : 0u,
            ac_mainline ? 1u : 0u,
            ae_mainline ? 1u : 0u,
            af_mainline ? 1u : 0u);
        std::printf(
            "CASE_%s_LID0_ATTN_FALLBACK_FLAGS p11ad_q_fallback_taken=%u p11ac_fallback_taken=%u p11ae_score_fallback_taken=%u p11af_softmax_output_fallback_taken=%u\n",
            case_tag,
            ad_fallback ? 1u : 0u,
            ac_fallback ? 1u : 0u,
            ae_fallback ? 1u : 0u,
            af_fallback ? 1u : 0u);
        if (emit_legacy_lines) {
            std::printf(
                "LID0_ATTN_MAINLINE_FLAGS p11ad_mainline_q_path_taken=%u p11ac_mainline_path_taken=%u p11ae_mainline_score_path_taken=%u p11af_mainline_softmax_output_path_taken=%u\n",
                ad_mainline ? 1u : 0u,
                ac_mainline ? 1u : 0u,
                ae_mainline ? 1u : 0u,
                af_mainline ? 1u : 0u);
            std::printf(
                "LID0_ATTN_FALLBACK_FLAGS p11ad_q_fallback_taken=%u p11ac_fallback_taken=%u p11ae_score_fallback_taken=%u p11af_softmax_output_fallback_taken=%u\n",
                ad_fallback ? 1u : 0u,
                ac_fallback ? 1u : 0u,
                ae_fallback ? 1u : 0u,
                af_fallback ? 1u : 0u);
        }

        if (!ad_mainline || !ac_mainline || !ae_mainline || !af_mainline) {
            std::printf(
                "[p11aj][FAIL] %s lid0 mainline flags invalid (ac=%d ad=%d ae=%d af=%d)\n",
                case_tag,
                ac_mainline ? 1 : 0,
                ad_mainline ? 1 : 0,
                ae_mainline ? 1 : 0,
                af_mainline ? 1 : 0);
            return false;
        }
        if (ad_fallback || ac_fallback || ae_fallback || af_fallback) {
            std::printf(
                "[p11aj][FAIL] %s lid0 fallback flag asserted (ac=%d ad=%d ae=%d af=%d)\n",
                case_tag,
                ac_fallback ? 1 : 0,
                ad_fallback ? 1 : 0,
                ae_fallback ? 1 : 0,
                af_fallback ? 1 : 0);
            return false;
        }

        std::printf("CASE_%s_LID0_ATTN_STAGE_AD_MAINLINE_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AC_MAINLINE_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AE_MAINLINE_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AF_MAINLINE_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AD_FALLBACK_NOT_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AC_FALLBACK_NOT_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AE_FALLBACK_NOT_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_STAGE_AF_FALLBACK_NOT_TAKEN PASS\n", case_tag);
        std::printf("CASE_%s_LID0_ATTN_DIRECT_SRAM_FALLBACK_NOT_TAKEN PASS\n", case_tag);
        if (emit_legacy_lines) {
            std::printf("LID0_ATTN_STAGE_AD_MAINLINE_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AC_MAINLINE_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AE_MAINLINE_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AF_MAINLINE_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AD_FALLBACK_NOT_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AC_FALLBACK_NOT_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AE_FALLBACK_NOT_TAKEN PASS\n");
            std::printf("LID0_ATTN_STAGE_AF_FALLBACK_NOT_TAKEN PASS\n");
            std::printf("LID0_ATTN_DIRECT_SRAM_FALLBACK_NOT_TAKEN PASS\n");
        }
        return true;
    }

    bool check_handoff_counter_conservation(
        const aecct::TopRegs& regs,
        const char* case_tag
    ) const {
        struct CounterTriplet {
            const char* name;
            uint32_t gate;
            uint32_t non_empty;
            uint32_t fallback;
        };
        const CounterTriplet checks[] = {
            {"p11av_ffn_handoff", (uint32_t)regs.p11av_ffn_handoff_gate_taken_count.to_uint(),
                (uint32_t)regs.p11av_ffn_handoff_non_empty_count.to_uint(),
                (uint32_t)regs.p11av_ffn_handoff_fallback_seen_count.to_uint()},
            {"p11ax_attn_out_payload", (uint32_t)regs.p11ax_attn_out_payload_gate_taken_count.to_uint(),
                (uint32_t)regs.p11ax_attn_out_payload_non_empty_count.to_uint(),
                (uint32_t)regs.p11ax_attn_out_payload_fallback_seen_count.to_uint()},
            {"p11ay_qkscore_mask", (uint32_t)regs.p11ay_qkscore_mask_handoff_gate_taken_count.to_uint(),
                (uint32_t)regs.p11ay_qkscore_mask_handoff_non_empty_count.to_uint(),
                (uint32_t)regs.p11ay_qkscore_mask_handoff_fallback_seen_count.to_uint()},
            {"p11az_qkscore_kvscan", (uint32_t)regs.p11az_qkscore_kvscan_handoff_gate_taken_count.to_uint(),
                (uint32_t)regs.p11az_qkscore_kvscan_handoff_non_empty_count.to_uint(),
                (uint32_t)regs.p11az_qkscore_kvscan_handoff_fallback_seen_count.to_uint()},
            {"p11ba_qkscore_qsrc", (uint32_t)regs.p11ba_qkscore_qsrc_handoff_gate_taken_count.to_uint(),
                (uint32_t)regs.p11ba_qkscore_qsrc_handoff_non_empty_count.to_uint(),
                (uint32_t)regs.p11ba_qkscore_qsrc_handoff_fallback_seen_count.to_uint()},
            {"p11bb_qkscore_wq", (uint32_t)regs.p11bb_qkscore_wq_handoff_gate_taken_count.to_uint(),
                (uint32_t)regs.p11bb_qkscore_wq_handoff_non_empty_count.to_uint(),
                (uint32_t)regs.p11bb_qkscore_wq_handoff_fallback_seen_count.to_uint()}
        };

        for (const auto& c : checks) {
            if (c.gate != (c.non_empty + c.fallback)) {
                std::printf(
                    "[p11aj][FAIL] %s counter conservation mismatch %s gate=%u non_empty=%u fallback=%u\n",
                    case_tag,
                    c.name,
                    (unsigned)c.gate,
                    (unsigned)c.non_empty,
                    (unsigned)c.fallback);
                return false;
            }
        }
        std::printf("CASE_%s_HANDOFF_COUNTER_CONSERVATION PASS\n", case_tag);
        return true;
    }

    bool init_state() {
        token_count_ = (uint32_t)aecct::ATTN_TOKEN_COUNT;
        d_model_ = (uint32_t)aecct::ATTN_D_MODEL;
        n_heads_ = (uint32_t)aecct::ATTN_N_HEADS;
        d_head_ = (uint32_t)aecct::ATTN_D_HEAD;
        param_base_ = (uint32_t)sram_map::W_REGION_BASE;
        staged_final_x_base_ = 0u;
        ae_any_score_change_ = false;
        af_any_pre_change_ = false;
        af_any_post_change_ = false;
        af_any_out_change_ = false;

        if (!p11aeaf_tb::prepare_qkv_payload_set(payloads_)) {
            std::printf("[p11aj][FAIL] payload preparation failed\n");
            return false;
        }

        seed_full_loop_sram(sram_stage_);
        seed_full_loop_sram(sram_full_);

        cfg_ = p11aeaf_tb::build_cfg();
        cfg_.n_layers = (aecct::u32_t)1u;
        sc_ = aecct::make_layer_scratch((aecct::u32_t)aecct::LN_X_OUT_BASE_WORD);

        init_top_regs_for_layers(regs_full_, 1u);
        return true;
    }

    bool run_stage_ac_provenance() {
        const uint32_t tensor_words = token_count_ * d_model_;
        const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t k_act_q_base = (uint32_t)sc_.attn.k_act_q_base_word.to_uint();
        const uint32_t v_act_q_base = (uint32_t)sc_.attn.v_act_q_base_word.to_uint();

        const std::vector<aecct::u32_t> before = sram_stage_;
        const aecct::LayerParamBase pb =
            aecct::make_layer_param_base((aecct::u32_t)param_base_, (aecct::u32_t)0u);
        bool fallback_taken = true;
        const bool mainline_taken = aecct::run_p11ac_layer0_top_managed_kv(
            sram_stage_.data(),
            cfg_,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            sc_,
            pb,
            fallback_taken);
        if (!mainline_taken || fallback_taken) {
            std::printf("[p11aj][FAIL] AC mainline/fallback status invalid (mainline=%d fallback=%d)\n",
                mainline_taken ? 1 : 0, fallback_taken ? 1 : 0);
            return false;
        }

        const uint32_t changed_k = count_span_changes(before, sram_stage_, k_base, tensor_words);
        const uint32_t changed_v = count_span_changes(before, sram_stage_, v_base, tensor_words);
        const uint32_t changed_k_act_q = count_span_changes(before, sram_stage_, k_act_q_base, tensor_words);
        const uint32_t changed_v_act_q = count_span_changes(before, sram_stage_, v_act_q_base, tensor_words);
        if (changed_k == 0u || changed_v == 0u || changed_k_act_q == 0u || changed_v_act_q == 0u) {
            std::printf("[p11aj][FAIL] AC target spans not written (K=%u V=%u K_act_q=%u V_act_q=%u)\n",
                (unsigned)changed_k, (unsigned)changed_v, (unsigned)changed_k_act_q, (unsigned)changed_v_act_q);
            return false;
        }

        if (!span_unchanged(before, sram_stage_, x_base, tensor_words, "AC source X")) {
            return false;
        }

        std::vector<uint8_t> allowed((uint32_t)sram_stage_.size(), 0u);
        mark_span(allowed, k_base, tensor_words);
        mark_span(allowed, v_base, tensor_words);
        mark_span(allowed, k_act_q_base, tensor_words);
        mark_span(allowed, v_act_q_base, tensor_words);
        if (!check_no_spurious_changes(before, sram_stage_, allowed, "AC")) {
            return false;
        }

        std::printf("PROVENANCE_STAGE_AC PASS\n");
        std::printf("PROVENANCE_STAGE_AC_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool run_stage_ad_provenance() {
        const uint32_t tensor_words = token_count_ * d_model_;
        const uint32_t x_base = (uint32_t)aecct::LN_X_OUT_BASE_WORD;
        const uint32_t q_base = (uint32_t)sc_.attn.q_base_word.to_uint();
        const uint32_t q_act_q_base = (uint32_t)sc_.attn.q_act_q_base_word.to_uint();
        const uint32_t q_sx_base = (uint32_t)sc_.attn.q_sx_base_word.to_uint();

        const std::vector<aecct::u32_t> before = sram_stage_;
        const aecct::LayerParamBase pb =
            aecct::make_layer_param_base((aecct::u32_t)param_base_, (aecct::u32_t)0u);
        bool fallback_taken = true;
        const bool mainline_taken = aecct::run_p11ad_layer0_top_managed_q(
            sram_stage_.data(),
            cfg_,
            (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD,
            sc_,
            pb,
            fallback_taken);
        if (!mainline_taken || fallback_taken) {
            std::printf("[p11aj][FAIL] AD mainline/fallback status invalid (mainline=%d fallback=%d)\n",
                mainline_taken ? 1 : 0, fallback_taken ? 1 : 0);
            return false;
        }

        const uint32_t changed_q = count_span_changes(before, sram_stage_, q_base, tensor_words);
        const uint32_t changed_q_act_q = count_span_changes(before, sram_stage_, q_act_q_base, tensor_words);
        const uint32_t changed_q_sx = count_span_changes(before, sram_stage_, q_sx_base, 1u);
        if (changed_q == 0u || changed_q_act_q == 0u || changed_q_sx == 0u) {
            std::printf("[p11aj][FAIL] AD target spans not written (Q=%u Q_act_q=%u Q_sx=%u)\n",
                (unsigned)changed_q, (unsigned)changed_q_act_q, (unsigned)changed_q_sx);
            return false;
        }

        if (!span_unchanged(before, sram_stage_, x_base, tensor_words, "AD source X")) {
            return false;
        }

        std::vector<uint8_t> allowed((uint32_t)sram_stage_.size(), 0u);
        mark_span(allowed, q_base, tensor_words);
        mark_span(allowed, q_act_q_base, tensor_words);
        mark_span(allowed, q_sx_base, 1u);
        if (!check_no_spurious_changes(before, sram_stage_, allowed, "AD")) {
            return false;
        }

        std::printf("PROVENANCE_STAGE_AD PASS\n");
        std::printf("PROVENANCE_STAGE_AD_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    bool prove_ae_consumes_qk(uint32_t token_idx) {
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t score_words = n_heads_ * token_count_;
        const uint32_t q_base = (uint32_t)sc_.attn.q_base_word.to_uint();
        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();
        const uint32_t q_addr = q_base + token_idx * d_model_;
        const uint32_t k_addr = k_base + token_idx * d_model_;

        std::vector<aecct::u32_t> sram_baseline = sram_stage_;
        std::vector<aecct::u32_t> sram_q_perturb = sram_stage_;
        std::vector<aecct::u32_t> sram_k_perturb = sram_stage_;

        bool fb_baseline = true;
        if (!aecct::run_p11ae_layer0_top_managed_qk_score(
                sram_baseline.data(), cfg_, sc_, (aecct::u32_t)token_idx, fb_baseline) || fb_baseline) {
            std::printf("[p11aj][FAIL] baseline AE execution failed for bridge proof\n");
            return false;
        }

        sram_q_perturb[q_addr] = force_delta_bits(sram_q_perturb[q_addr]);
        bool fb_q = true;
        if (!aecct::run_p11ae_layer0_top_managed_qk_score(
                sram_q_perturb.data(), cfg_, sc_, (aecct::u32_t)token_idx, fb_q) || fb_q) {
            std::printf("[p11aj][FAIL] Q-perturbed AE execution failed for bridge proof\n");
            return false;
        }

        sram_k_perturb[k_addr] = force_delta_bits(sram_k_perturb[k_addr]);
        bool fb_k = true;
        if (!aecct::run_p11ae_layer0_top_managed_qk_score(
                sram_k_perturb.data(), cfg_, sc_, (aecct::u32_t)token_idx, fb_k) || fb_k) {
            std::printf("[p11aj][FAIL] K-perturbed AE execution failed for bridge proof\n");
            return false;
        }

        const uint32_t q_bridge_diffs = count_span_diffs(sram_baseline, sram_q_perturb, score_base, score_words);
        const uint32_t k_bridge_diffs = count_span_diffs(sram_baseline, sram_k_perturb, score_base, score_words);
        if (q_bridge_diffs == 0u) {
            std::printf("[p11aj][FAIL] Q perturbation did not change score span\n");
            return false;
        }
        if (k_bridge_diffs == 0u) {
            std::printf("[p11aj][FAIL] K perturbation did not change score span\n");
            return false;
        }

        std::printf("BRIDGE_Q_TO_SCORE_CONSUMPTION PASS\n");
        std::printf("BRIDGE_K_TO_SCORE_CONSUMPTION PASS\n");
        return true;
    }

    bool run_stage_ae_provenance(uint32_t token_idx) {
        const uint32_t tensor_words = token_count_ * d_model_;
        const uint32_t score_words = n_heads_ * token_count_;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t q_base = (uint32_t)sc_.attn.q_base_word.to_uint();
        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();

        std::vector<aecct::u32_t> expected_score;
        p11aeaf_tb::compute_expected_score_row(
            sram_stage_,
            sc_.attn,
            token_idx,
            token_count_,
            n_heads_,
            d_head_,
            expected_score);

        const std::vector<aecct::u32_t> before = sram_stage_;
        bool fallback_taken = true;
        const bool mainline_taken = aecct::run_p11ae_layer0_top_managed_qk_score(
            sram_stage_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken);
        if (!mainline_taken || fallback_taken) {
            std::printf("[p11aj][FAIL] AE mainline/fallback status invalid token=%u (mainline=%d fallback=%d)\n",
                (unsigned)token_idx, mainline_taken ? 1 : 0, fallback_taken ? 1 : 0);
            return false;
        }

        for (uint32_t i = 0u; i < score_words; ++i) {
            const uint32_t got = (uint32_t)sram_stage_[score_base + i].to_uint();
            const uint32_t exp = (uint32_t)expected_score[i].to_uint();
            if (got != exp) {
                std::printf("[p11aj][FAIL] AE expected score mismatch token=%u idx=%u got=0x%08X exp=0x%08X\n",
                    (unsigned)token_idx, (unsigned)i, (unsigned)got, (unsigned)exp);
                return false;
            }
        }

        const uint32_t changed_score = count_span_changes(before, sram_stage_, score_base, score_words);
        if (changed_score != 0u) {
            ae_any_score_change_ = true;
        }

        if (!span_unchanged(before, sram_stage_, q_base, tensor_words, "AE source Q")) {
            return false;
        }
        if (!span_unchanged(before, sram_stage_, k_base, tensor_words, "AE source K")) {
            return false;
        }

        std::vector<uint8_t> allowed((uint32_t)sram_stage_.size(), 0u);
        mark_span(allowed, score_base, score_words);
        if (!check_no_spurious_changes(before, sram_stage_, allowed, "AE")) {
            return false;
        }
        return true;
    }

    bool prove_af_consumes_score(uint32_t token_idx) {
        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx * d_model_;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();

        std::vector<aecct::u32_t> sram_baseline = sram_stage_;
        std::vector<aecct::u32_t> sram_score_perturb = sram_stage_;

        bool fb_baseline = true;
        if (!aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_baseline.data(), cfg_, sc_, (aecct::u32_t)token_idx, fb_baseline) || fb_baseline) {
            std::printf("[p11aj][FAIL] baseline AF execution failed for bridge proof\n");
            return false;
        }

        sram_score_perturb[score_base] = force_delta_bits(sram_score_perturb[score_base]);
        bool fb_perturb = true;
        if (!aecct::run_p11af_layer0_top_managed_softmax_out(
                sram_score_perturb.data(), cfg_, sc_, (aecct::u32_t)token_idx, fb_perturb) || fb_perturb) {
            std::printf("[p11aj][FAIL] score-perturbed AF execution failed for bridge proof\n");
            return false;
        }

        const uint32_t out_diffs = count_span_diffs(sram_baseline, sram_score_perturb, out_row_base, d_model_);
        if (out_diffs == 0u) {
            std::printf("[p11aj][FAIL] score perturbation did not change AF output row\n");
            return false;
        }

        std::printf("BRIDGE_SCORE_TO_OUTPUT_CONSUMPTION PASS\n");
        return true;
    }

    bool run_stage_af_provenance(uint32_t token_idx) {
        const uint32_t tensor_words = token_count_ * d_model_;
        const uint32_t score_words = n_heads_ * token_count_;
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t pre_row_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint() + token_idx * d_model_;
        const uint32_t post_row_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint() + token_idx * d_model_;
        const uint32_t out_row_base = (uint32_t)sc_.attn_out_base_word.to_uint() + token_idx * d_model_;

        std::vector<aecct::u32_t> expected_out;
        p11aeaf_tb::compute_expected_output_row_online(
            sram_stage_,
            sc_.attn,
            token_idx,
            token_count_,
            n_heads_,
            d_head_,
            expected_out);

        const std::vector<aecct::u32_t> before = sram_stage_;
        bool fallback_taken = true;
        const bool mainline_taken = aecct::run_p11af_layer0_top_managed_softmax_out(
            sram_stage_.data(),
            cfg_,
            sc_,
            (aecct::u32_t)token_idx,
            fallback_taken);
        if (!mainline_taken || fallback_taken) {
            std::printf("[p11aj][FAIL] AF mainline/fallback status invalid token=%u (mainline=%d fallback=%d)\n",
                (unsigned)token_idx, mainline_taken ? 1 : 0, fallback_taken ? 1 : 0);
            return false;
        }

        for (uint32_t i = 0u; i < d_model_; ++i) {
            const uint32_t exp = (uint32_t)expected_out[i].to_uint();
            const uint32_t got_pre = (uint32_t)sram_stage_[pre_row_base + i].to_uint();
            const uint32_t got_post = (uint32_t)sram_stage_[post_row_base + i].to_uint();
            const uint32_t got_out = (uint32_t)sram_stage_[out_row_base + i].to_uint();
            if (got_pre != exp || got_post != exp || got_out != exp) {
                std::printf(
                    "[p11aj][FAIL] AF expected output mismatch token=%u idx=%u pre=0x%08X post=0x%08X out=0x%08X exp=0x%08X\n",
                    (unsigned)token_idx, (unsigned)i,
                    (unsigned)got_pre, (unsigned)got_post, (unsigned)got_out, (unsigned)exp);
                return false;
            }
        }

        const uint32_t changed_pre = count_span_changes(before, sram_stage_, pre_row_base, d_model_);
        const uint32_t changed_post = count_span_changes(before, sram_stage_, post_row_base, d_model_);
        const uint32_t changed_out = count_span_changes(before, sram_stage_, out_row_base, d_model_);
        if (changed_pre != 0u) { af_any_pre_change_ = true; }
        if (changed_post != 0u) { af_any_post_change_ = true; }
        if (changed_out != 0u) { af_any_out_change_ = true; }

        if (!span_unchanged(before, sram_stage_, score_base, score_words, "AF source SCORE")) {
            return false;
        }
        if (!span_unchanged(before, sram_stage_, v_base, tensor_words, "AF source V")) {
            return false;
        }

        std::vector<uint8_t> allowed((uint32_t)sram_stage_.size(), 0u);
        mark_span(allowed, pre_row_base, d_model_);
        mark_span(allowed, post_row_base, d_model_);
        mark_span(allowed, out_row_base, d_model_);
        if (!check_no_spurious_changes(before, sram_stage_, allowed, "AF")) {
            return false;
        }
        return true;
    }

    bool run_staged_provenance_chain() {
        if (!run_stage_ad_provenance()) {
            return false;
        }
        if (!run_stage_ac_provenance()) {
            return false;
        }
        if (!prove_ae_consumes_qk(0u)) {
            return false;
        }

        for (uint32_t t = 0u; t < token_count_; ++t) {
            if (!run_stage_ae_provenance(t)) {
                return false;
            }
            if (t == 0u) {
                if (!prove_af_consumes_score(t)) {
                    return false;
                }
            }
            if (!run_stage_af_provenance(t)) {
                return false;
            }
        }

        if (!ae_any_score_change_) {
            std::printf("[p11aj][FAIL] AE stage never produced an observable score-span change\n");
            return false;
        }
        if (!af_any_pre_change_ || !af_any_post_change_ || !af_any_out_change_) {
            std::printf("[p11aj][FAIL] AF stage did not produce observable target-span changes (pre=%d post=%d out=%d)\n",
                af_any_pre_change_ ? 1 : 0,
                af_any_post_change_ ? 1 : 0,
                af_any_out_change_ ? 1 : 0);
            return false;
        }

        std::printf("PROVENANCE_STAGE_AE PASS\n");
        std::printf("PROVENANCE_STAGE_AE_NO_SPURIOUS_TOUCH PASS\n");
        std::printf("PROVENANCE_STAGE_AF PASS\n");
        std::printf("PROVENANCE_STAGE_AF_NO_SPURIOUS_TOUCH PASS\n");
        return true;
    }

    uint32_t run_downstream_from_prebuilt(std::vector<aecct::u32_t>& sram_vec) const {
        aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD;
        aecct::u32_t x_out_base = aecct::alternate_x_page(x_in_base);
        const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
        const aecct::LayerParamBase pb =
            aecct::make_layer_param_base((aecct::u32_t)param_base_, (aecct::u32_t)0u);

        aecct::TransformerLayer(
            sram_vec.data(),
            cfg_,
            (aecct::u32_t)0u,
            x_in_base,
            x_out_base,
            sc,
            pb,
            true,  // kv_prebuilt_from_top_managed
            true,  // q_prebuilt_from_top_managed
            true,  // score_prebuilt_from_top_managed
            true   // out_prebuilt_from_top_managed
        );

        x_in_base = x_out_base;
        x_out_base = aecct::alternate_x_page(x_in_base);
        aecct::run_mid_or_end_layernorm(
            false,
            cfg_,
            sram_vec.data(),
            (aecct::u32_t)param_base_,
            x_in_base,
            x_out_base
        );

        x_in_base = x_out_base;
        return (uint32_t)x_in_base.to_uint();
    }

    uint32_t run_transformer_layer_prebuilt_only(std::vector<aecct::u32_t>& sram_vec) const {
        aecct::u32_t x_in_base = (aecct::u32_t)aecct::LN_X_OUT_BASE_WORD;
        aecct::u32_t x_out_base = aecct::alternate_x_page(x_in_base);
        const aecct::LayerScratch sc = aecct::make_layer_scratch(x_in_base);
        const aecct::LayerParamBase pb =
            aecct::make_layer_param_base((aecct::u32_t)param_base_, (aecct::u32_t)0u);

        aecct::TransformerLayer(
            sram_vec.data(),
            cfg_,
            (aecct::u32_t)0u,
            x_in_base,
            x_out_base,
            sc,
            pb,
            true,  // kv_prebuilt_from_top_managed
            true,  // q_prebuilt_from_top_managed
            true,  // score_prebuilt_from_top_managed
            true   // out_prebuilt_from_top_managed
        );
        return (uint32_t)sc.ffn.add2_base_word.to_uint();
    }

    bool run_staged_downstream_finalize() {
        std::vector<aecct::u32_t> sram_baseline = sram_stage_;
        std::vector<aecct::u32_t> sram_perturb = sram_stage_;
        const uint32_t attn_out_base = (uint32_t)sc_.attn_out_base_word.to_uint();
        sram_perturb[attn_out_base] = force_delta_bits(sram_perturb[attn_out_base]);

        const uint32_t add2_base_baseline = run_transformer_layer_prebuilt_only(sram_baseline);
        const uint32_t add2_base_perturb = run_transformer_layer_prebuilt_only(sram_perturb);
        if (add2_base_baseline != add2_base_perturb) {
            std::printf("[p11aj][FAIL] downstream add2 base mismatch baseline=%u perturb=%u\n",
                (unsigned)add2_base_baseline, (unsigned)add2_base_perturb);
            return false;
        }
        const uint32_t add2_words = token_count_ * d_model_;
        const uint32_t add2_diffs =
            count_span_diffs(sram_baseline, sram_perturb, add2_base_baseline, add2_words);
        if (add2_diffs == 0u) {
            std::printf("[p11aj][FAIL] attn_out perturbation did not propagate to downstream add2 span\n");
            return false;
        }

        staged_final_x_base_ = run_downstream_from_prebuilt(sram_stage_);
        sram_stage_before_downstream_ = sram_stage_;

        std::printf("BRIDGE_ATTNOUT_TO_DOWNSTREAM_CONSUMPTION PASS\n");
        return true;
    }

    bool run_direct_attnout_to_finalx_bridge_probe() {
        const uint32_t attn_out_base = (uint32_t)sc_.attn_out_base_word.to_uint();
        const uint32_t final_words = (uint32_t)aecct::LN_X_TOTAL_WORDS;
        const uint32_t add2_words = token_count_ * d_model_;

        std::vector<aecct::u32_t> sram_add2_baseline = sram_stage_;
        std::vector<aecct::u32_t> sram_add2_perturb = sram_stage_;
        apply_bridge_probe_norm_params(sram_add2_baseline);
        apply_bridge_probe_norm_params(sram_add2_perturb);
        sram_add2_perturb[attn_out_base] = force_delta_bits(sram_add2_perturb[attn_out_base]);

        const uint32_t add2_base_baseline = run_transformer_layer_prebuilt_only(sram_add2_baseline);
        const uint32_t add2_base_perturb = run_transformer_layer_prebuilt_only(sram_add2_perturb);
        if (add2_base_baseline != add2_base_perturb) {
            std::printf("[p11ak][FAIL] add2 base mismatch baseline=%u perturb=%u\n",
                (unsigned)add2_base_baseline, (unsigned)add2_base_perturb);
            return false;
        }
        const uint32_t add2_diffs =
            count_span_diffs(sram_add2_baseline, sram_add2_perturb, add2_base_baseline, add2_words);
        if (add2_diffs == 0u) {
            std::printf("[p11ak][FAIL] attn_out perturbation did not change add2 span in direct probe\n");
            return false;
        }

        std::vector<aecct::u32_t> sram_final_baseline = sram_stage_;
        std::vector<aecct::u32_t> sram_final_perturb = sram_stage_;
        apply_bridge_probe_norm_params(sram_final_baseline);
        apply_bridge_probe_norm_params(sram_final_perturb);
        sram_final_perturb[attn_out_base] = force_delta_bits(sram_final_perturb[attn_out_base]);

        const uint32_t final_base_baseline = run_downstream_from_prebuilt(sram_final_baseline);
        const uint32_t final_base_perturb = run_downstream_from_prebuilt(sram_final_perturb);
        if (final_base_baseline != final_base_perturb) {
            std::printf("[p11ak][FAIL] final_x base mismatch baseline=%u perturb=%u\n",
                (unsigned)final_base_baseline, (unsigned)final_base_perturb);
            return false;
        }
        const uint32_t final_diffs =
            count_span_diffs(sram_final_baseline, sram_final_perturb, final_base_baseline, final_words);
        if (final_diffs == 0u) {
            std::printf("[p11ak][FAIL] attn_out perturbation did not change final_x span in direct probe\n");
            return false;
        }

        std::printf("[p11ak][DIRECT_BRIDGE][PASS] add2_diffs=%u final_x_diffs=%u\n",
            (unsigned)add2_diffs, (unsigned)final_diffs);
        std::printf("BRIDGE_ATTNOUT_TO_FINALX_DIRECT_CONSUMPTION PASS\n");
        return true;
    }

    bool run_full_loop_mainline() {
        aecct::run_transformer_layer_loop(regs_full_, sram_full_.data());
        if (!check_lid0_marker_contract(regs_full_, "BASELINE_N1", true)) {
            return false;
        }
        if (!check_handoff_counter_conservation(regs_full_, "BASELINE_N1")) {
            return false;
        }
        std::printf("FULL_LOOP_MAINLINE_PATH_TAKEN PASS\n");
        std::printf("fallback_taken = false\n");
        std::printf("FULL_LOOP_FALLBACK_NOT_TAKEN PASS\n");
        return true;
    }

    bool run_full_loop_mixed_layer_mainline() {
        std::vector<aecct::u32_t> sram_mixed_a;
        std::vector<aecct::u32_t> sram_mixed_b;
        seed_full_loop_sram(sram_mixed_a);
        seed_full_loop_sram(sram_mixed_b);

        aecct::TopRegs regs_mixed_a;
        aecct::TopRegs regs_mixed_b;
        init_top_regs_for_layers(regs_mixed_a, 3u);
        init_top_regs_for_layers(regs_mixed_b, 3u);

        aecct::run_transformer_layer_loop(regs_mixed_a, sram_mixed_a.data());
        aecct::run_transformer_layer_loop(regs_mixed_b, sram_mixed_b.data());

        if (!check_lid0_marker_contract(regs_mixed_a, "MIXED_N3", false)) {
            return false;
        }
        if (!check_handoff_counter_conservation(regs_mixed_a, "MIXED_N3")) {
            return false;
        }

        if (regs_mixed_a.p11ad_mainline_q_path_taken != regs_full_.p11ad_mainline_q_path_taken ||
            regs_mixed_a.p11ac_mainline_path_taken != regs_full_.p11ac_mainline_path_taken ||
            regs_mixed_a.p11ae_mainline_score_path_taken != regs_full_.p11ae_mainline_score_path_taken ||
            regs_mixed_a.p11af_mainline_softmax_output_path_taken != regs_full_.p11af_mainline_softmax_output_path_taken ||
            regs_mixed_a.p11ad_q_fallback_taken != regs_full_.p11ad_q_fallback_taken ||
            regs_mixed_a.p11ac_fallback_taken != regs_full_.p11ac_fallback_taken ||
            regs_mixed_a.p11ae_score_fallback_taken != regs_full_.p11ae_score_fallback_taken ||
            regs_mixed_a.p11af_softmax_output_fallback_taken != regs_full_.p11af_softmax_output_fallback_taken) {
            std::printf("[p11aj][FAIL] mixed n_layers=3 lid0 marker mismatch vs baseline n_layers=1 snapshot\n");
            return false;
        }
        std::printf("CASE_MIXED_N3_LID0_MARKERS_STABLE_VS_BASELINE PASS\n");

        if (regs_mixed_a.p11ad_mainline_q_path_taken != regs_mixed_b.p11ad_mainline_q_path_taken ||
            regs_mixed_a.p11ac_mainline_path_taken != regs_mixed_b.p11ac_mainline_path_taken ||
            regs_mixed_a.p11ae_mainline_score_path_taken != regs_mixed_b.p11ae_mainline_score_path_taken ||
            regs_mixed_a.p11af_mainline_softmax_output_path_taken != regs_mixed_b.p11af_mainline_softmax_output_path_taken ||
            regs_mixed_a.p11ad_q_fallback_taken != regs_mixed_b.p11ad_q_fallback_taken ||
            regs_mixed_a.p11ac_fallback_taken != regs_mixed_b.p11ac_fallback_taken ||
            regs_mixed_a.p11ae_score_fallback_taken != regs_mixed_b.p11ae_score_fallback_taken ||
            regs_mixed_a.p11af_softmax_output_fallback_taken != regs_mixed_b.p11af_softmax_output_fallback_taken) {
            std::printf("[p11aj][FAIL] mixed n_layers=3 lid0 marker mismatch between repeated runs\n");
            return false;
        }
        std::printf("CASE_MIXED_N3_MARKER_REPEATABILITY PASS\n");

        const uint32_t final_base_a = (uint32_t)regs_mixed_a.infer_final_x_base_word.to_uint();
        const uint32_t final_base_b = (uint32_t)regs_mixed_b.infer_final_x_base_word.to_uint();
        if (final_base_a != final_base_b) {
            std::printf("[p11aj][FAIL] mixed n_layers=3 final_x base mismatch runA=%u runB=%u\n",
                (unsigned)final_base_a,
                (unsigned)final_base_b);
            return false;
        }
        if (!compare_span_exact(
                sram_mixed_a,
                sram_mixed_b,
                final_base_a,
                (uint32_t)aecct::LN_X_TOTAL_WORDS,
                "mixed_n3_final_x_repeatability")) {
            return false;
        }
        std::printf("CASE_MIXED_N3_FINAL_X_DETERMINISTIC PASS\n");

        for (uint32_t i = 0u; i < (uint32_t)aecct::LN_X_TOTAL_WORDS; ++i) {
            const uint32_t bits = (uint32_t)sram_mixed_a[final_base_a + i].to_uint();
            if (is_nonfinite_bits(bits)) {
                std::printf(
                    "[p11aj][FAIL] mixed n_layers=3 final_x non-finite at idx=%u bits=0x%08X\n",
                    (unsigned)i,
                    (unsigned)bits);
                return false;
            }
        }
        std::printf("CASE_MIXED_N3_FINAL_X_NONFINITE_SCAN PASS\n");

        const uint32_t baseline_final_base = (uint32_t)regs_full_.infer_final_x_base_word.to_uint();
        if (baseline_final_base == final_base_a) {
            const uint32_t diffs = count_span_diffs(
                sram_full_,
                sram_mixed_a,
                final_base_a,
                (uint32_t)aecct::LN_X_TOTAL_WORDS);
            std::printf("CASE_MIXED_N3_FINAL_X_DIFFS_VS_BASELINE_N1 = %u\n", (unsigned)diffs);
        } else {
            std::printf(
                "CASE_MIXED_N3_FINAL_X_BASE_NOTE baseline_base=%u mixed_base=%u\n",
                (unsigned)baseline_final_base,
                (unsigned)final_base_a);
        }

        std::printf("CASE_MIXED_N3_ACCEPTANCE PASS\n");
        return true;
    }

    bool validate_full_flow_key_span_compare() {
        const uint32_t tensor_words = token_count_ * d_model_;
        const uint32_t score_words = n_heads_ * token_count_;
        const uint32_t q_base = (uint32_t)sc_.attn.q_base_word.to_uint();
        const uint32_t k_base = (uint32_t)sc_.attn.k_base_word.to_uint();
        const uint32_t v_base = (uint32_t)sc_.attn.v_base_word.to_uint();
        const uint32_t q_act_q_base = (uint32_t)sc_.attn.q_act_q_base_word.to_uint();
        const uint32_t k_act_q_base = (uint32_t)sc_.attn.k_act_q_base_word.to_uint();
        const uint32_t v_act_q_base = (uint32_t)sc_.attn.v_act_q_base_word.to_uint();
        const uint32_t q_sx_base = (uint32_t)sc_.attn.q_sx_base_word.to_uint();
        const uint32_t score_base = (uint32_t)sc_.attn.score_base_word.to_uint();
        const uint32_t pre_base = (uint32_t)sc_.attn.pre_concat_base_word.to_uint();
        const uint32_t post_base = (uint32_t)sc_.attn.post_concat_base_word.to_uint();
        const uint32_t out_base = (uint32_t)sc_.attn_out_base_word.to_uint();

        if (!compare_span_exact(sram_full_, sram_stage_, q_base, tensor_words, "Q span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, k_base, tensor_words, "K span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, v_base, tensor_words, "V span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, q_act_q_base, tensor_words, "Q_act_q span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, k_act_q_base, tensor_words, "K_act_q span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, v_act_q_base, tensor_words, "V_act_q span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, q_sx_base, 1u, "Q_sx span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, score_base, score_words, "score span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, pre_base, tensor_words, "pre-concat span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, post_base, tensor_words, "post-concat span")) {
            return false;
        }
        if (!compare_span_exact(sram_full_, sram_stage_, out_base, tensor_words, "attn_out span")) {
            return false;
        }

        std::printf("FULL_FLOW_KEY_SPAN_EXPECTED_COMPARE PASS\n");
        return true;
    }

    bool validate_final_x_hardened_compare() {
        const uint32_t full_final_base = (uint32_t)regs_full_.infer_final_x_base_word.to_uint();
        if (staged_final_x_base_ != full_final_base) {
            std::printf("[p11aj][FAIL] final_x base mismatch staged=%u full=%u\n",
                (unsigned)staged_final_x_base_, (unsigned)full_final_base);
            return false;
        }

        for (uint32_t t = 0u; t < token_count_; ++t) {
            const uint32_t row_base = full_final_base + t * d_model_;
            for (uint32_t c = 0u; c < d_model_; ++c) {
                const uint32_t got = (uint32_t)sram_full_[row_base + c].to_uint();
                const uint32_t exp = (uint32_t)sram_stage_[row_base + c].to_uint();
                if (got != exp) {
                    std::printf("[p11aj][FAIL] final_x compare mismatch token=%u col=%u got=0x%08X exp=0x%08X\n",
                        (unsigned)t, (unsigned)c, (unsigned)got, (unsigned)exp);
                    return false;
                }
            }
        }

        std::printf("FINAL_X_EXPECTED_COMPARE_HARDENED PASS\n");
        return true;
    }
};

} // namespace

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbP11ajTopManagedSramProvenance tb;
    const int rc = tb.run_all();
    CCS_RETURN(rc);
}

#endif // __SYNTHESIS__
