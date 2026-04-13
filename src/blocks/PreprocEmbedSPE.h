#pragma once
// PreprocEmbedSPE clean rewrite bridge.
//
// Purpose of this round:
// - keep the original file name / public entry points so Top and existing TBs
//   still compile
// - move the active numeric path to fp16-first preproc composition
// - keep Top-owned SRAM / range contract explicit
// - allow a temporary header-backed weight provider while Top-loaded W_REGION
//   is still under bring-up
//
// Important boundary notes:
// - mainline clean compute uses fp16 activations and fp16 parameter reads
// - this file does NOT grant block-owned shared-SRAM arbitration
// - a small compatibility copy path is kept only for older pilot/checkpoint TBs
//   that exercise payload handoff rather than final preproc math

#include <cstdint>
#include <type_traits>

#include "AecctProtocol.h"
#include "AecctRanges.h"
#include "AecctTypes.h"
#include "AecctUtil.h"
#include "Fp16WeightProvider.h"
#include "PreprocDescBringup.h"
#include "XWorkU16HybridView.h"
#include "weights.h"

namespace aecct {

struct PreprocCfg {
    u32_t infer_in_words;
    u32_t x_out_words;
};

struct PreprocBlockContract {
    bool start;
    bool done;
    bool err_valid;
    u16_t err_code;
    TokenRange token_range;
    TileRange tile_range;
    PhaseId phase_id;
    u32_t x_work_base_word;
    u32_t w_base_word;
};

static inline void clear_preproc_contract(PreprocBlockContract& c) {
    c.start = false;
    c.done = false;
    c.err_valid = false;
    c.err_code = 0;
    c.token_range = make_token_range((u32_t)0u, (u32_t)0u);
    c.tile_range = make_tile_range((u32_t)0u, (u32_t)0u);
    c.phase_id = PHASE_PREPROC;
    c.x_work_base_word = (u32_t)0u;
    c.w_base_word = (u32_t)0u;
}

namespace preproc_clean {

struct PreprocFp16Debug {
    fp16_t var_feature[CODE_N];
    fp16_t check_feature[CODE_C];
    fp16_t node_feature[N_NODES];
    fp16_t preproc_x[N_NODES][D_MODEL];
    fp16_t x_work[N_NODES][D_MODEL];
};

static inline fp16_t preproc_fp16_from_double(double x) {
    return fp16_t(x);
}

static inline fp16_t preproc_fp16_abs(const fp16_t& x) {
    return (x < fp16_t(0.0)) ? fp16_t(-x.to_double()) : x;
}

static inline fp16_t preproc_mul_fp16(const fp16_t& a, const fp16_t& b) {
    // Match the current fp16 branch reference convention exactly:
    // - node feature is already fp16
    // - embed weight is already fp16
    // - multiply in fp16 expression form and round once back to fp16
    return fp16_t(a * b);
}

static inline fp16_t preproc_input_word_to_fp16(const u32_t& word_bits) {
    const fp32_t y_fp32 = fp32_from_bits(word_bits);
    return fp16_t(y_fp32);
}

static inline uint32_t preproc_token_begin(const PreprocBlockContract& contract) {
    return (uint32_t)contract.token_range.begin.to_uint();
}

static inline uint32_t preproc_token_end(const PreprocBlockContract& contract) {
    const uint32_t end = (uint32_t)contract.token_range.end.to_uint();
    if (end == 0u || end > (uint32_t)N_NODES) {
        return (uint32_t)N_NODES;
    }
    return end;
}

static inline bool use_clean_compute(const PreprocCfg& cfg) {
    return ((uint32_t)cfg.infer_in_words.to_uint() == (uint32_t)PREPROC_IN_WORDS_EXPECTED) &&
           ((uint32_t)cfg.x_out_words.to_uint() == (uint32_t)PREPROC_X_OUT_WORDS_EXPECTED);
}

static inline void run_preproc_fp16_clean_from_words(
    const u32_t input_words[CODE_N],
    const fp16_rewrite::Fp16PreprocWeightProvider& weights,
    PreprocFp16Debug& dbg) {
    PREPROC_VAR_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        const fp16_t y = preproc_input_word_to_fp16(input_words[v]);
        dbg.var_feature[v] = preproc_fp16_abs(y);
        dbg.node_feature[v] = dbg.var_feature[v];
    }

    PREPROC_CHECK_LOOP: for (uint32_t c = 0u; c < (uint32_t)CODE_C; ++c) {
        ac_int<1, false> parity = 0;
        PREPROC_CHECK_VAR_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
            const uint32_t flat = c * (uint32_t)CODE_N + v;
            if ((uint32_t)h_H[flat].to_uint() == 0u) {
                continue;
            }
            const fp16_t y = preproc_input_word_to_fp16(input_words[v]);
            if (y < fp16_t(0.0)) {
                parity = (ac_int<1, false>)((uint32_t)parity.to_uint() ^ 1u);
            }
        }
        dbg.check_feature[c] = ((uint32_t)parity.to_uint() == 0u) ? fp16_t(1.0) : fp16_t(-1.0);
        dbg.node_feature[(uint32_t)CODE_N + c] = dbg.check_feature[c];
    }

    PREPROC_TOKEN_LOOP: for (uint32_t t = 0u; t < (uint32_t)N_NODES; ++t) {
        PREPROC_SRC_LOOP: for (uint32_t k = 0u; k < (uint32_t)D_SRC_EMBED; ++k) {
            const fp16_t embed = weights.src_embed(t, k);
            const fp16_t mul = preproc_mul_fp16(dbg.node_feature[t], embed);
            dbg.preproc_x[t][k] = mul;
            dbg.x_work[t][k] = mul;
        }
        PREPROC_LPE_LOOP: for (uint32_t k = 0u; k < (uint32_t)D_LPE_TOKEN; ++k) {
            const fp16_t lpe = weights.lpe_token(t, k);
            const uint32_t d = (uint32_t)D_SRC_EMBED + k;
            dbg.preproc_x[t][d] = lpe;
            dbg.x_work[t][d] = lpe;
        }
    }
}

template<typename SramView>
static inline void x_work_store_fp16_bridge(
    SramView& sram,
    uint32_t x_base_word,
    uint32_t elem_idx,
    const fp16_t& value) {
    x_work_store_fp16(sram, x_base_word, elem_idx, value);
}

template<uint32_t MAIN_WORDS, uint32_t X_WORDS>
static inline void x_work_store_fp16_bridge(
    XWorkU16HybridView<MAIN_WORDS, X_WORDS>& view,
    uint32_t x_base_word,
    uint32_t elem_idx,
    const fp16_t& value) {
    const uint32_t addr = x_base_word + elem_idx;
    if (addr < X_WORDS) {
        view.x_work_words[addr] = bits_from_fp16(value);
    }
}

template<typename SramView>
static inline void write_clean_x_window(
    SramView& sram,
    uint32_t x_base_word,
    const PreprocBlockContract& contract,
    const PreprocFp16Debug& dbg) {
    const uint32_t token_begin = preproc_token_begin(contract);
    const uint32_t token_end = preproc_token_end(contract);
    PREPROC_WRITE_TOKEN_LOOP: for (uint32_t t = token_begin; t < token_end; ++t) {
        PREPROC_WRITE_DIM_LOOP: for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            const uint32_t elem_idx = t * (uint32_t)D_MODEL + d;
            x_work_store_fp16_bridge(sram, x_base_word, elem_idx, dbg.x_work[t][d]);
        }
    }
}

static inline void checkpoint_copy_u32(
    u32_t* sram,
    const PreprocCfg& cfg,
    const u32_t in_base_word,
    const u32_t x_base_word,
    const u32_t* topfed_in_payload) {
    const uint32_t in_words = (uint32_t)cfg.infer_in_words.to_uint();
    const uint32_t x_words = (uint32_t)cfg.x_out_words.to_uint();
    const uint32_t copy_words = (in_words < x_words) ? in_words : x_words;
    const uint32_t in_base = (uint32_t)in_base_word.to_uint();
    const uint32_t x_base = (uint32_t)x_base_word.to_uint();
    PREPROC_COPY_LOOP: for (uint32_t i = 0u; i < copy_words; ++i) {
        sram[x_base + i] = (topfed_in_payload != nullptr) ? topfed_in_payload[i] : sram[in_base + i];
    }
    PREPROC_COPY_ZERO_LOOP: for (uint32_t i = copy_words; i < x_words; ++i) {
        sram[x_base + i] = (u32_t)0u;
    }
}

} // namespace preproc_clean

static inline void PreprocEmbedSPE(
    u32_t* sram,
    const PreprocCfg& cfg,
    const u32_t in_base_word,
    const u32_t x_base_word) {
    // Legacy checkpoint wrapper kept for old M7-style smoke.
    preproc_clean::checkpoint_copy_u32(sram, cfg, in_base_word, x_base_word, nullptr);
}

template<typename SramView>
static inline void PreprocEmbedSPECoreWindow(
    SramView& sram,
    const PreprocCfg& cfg,
    const u32_t in_base_word,
    const u32_t x_base_word,
    PreprocBlockContract& contract,
    const u32_t* topfed_in_payload = nullptr) {
    contract.done = false;
    contract.err_valid = false;
    contract.err_code = 0;

    if (!contract.start) {
        contract.err_valid = true;
        contract.err_code = (u16_t)ERR_BAD_STATE;
        return;
    }

    if (!preproc_clean::use_clean_compute(cfg)) {
        if constexpr (std::is_same_v<std::remove_cv_t<std::remove_pointer_t<SramView>>, u32_t>) {
            preproc_clean::checkpoint_copy_u32(sram, cfg, in_base_word, x_base_word, topfed_in_payload);
            contract.done = true;
            return;
        }
    }

    u32_t input_words[CODE_N];
    const uint32_t in_base = (uint32_t)in_base_word.to_uint();
    PREPROC_INPUT_GATHER_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        if (topfed_in_payload != nullptr) {
            input_words[v] = topfed_in_payload[v];
        } else {
            input_words[v] = sram[in_base + v];
        }
    }

    fp16_rewrite::HeaderFp16PreprocWeightProvider weights;
    preproc_clean::PreprocFp16Debug dbg;
    preproc_clean::run_preproc_fp16_clean_from_words(input_words, weights, dbg);
    preproc_clean::write_clean_x_window(
        sram,
        (uint32_t)x_base_word.to_uint(),
        contract,
        dbg);
    contract.done = true;
}

static inline void PreprocEmbedSPEWord16(
    u16_t* sram_word16,
    const PreprocCfg& cfg,
    const u32_t in_base_word16,
    const u32_t x_base_word16,
    const u32_t* topfed_in_payload,
    const u32_t w_base_word16) {
    (void)sram_word16;
    (void)in_base_word16;
    (void)w_base_word16;

    PreprocBlockContract contract;
    clear_preproc_contract(contract);
    contract.start = true;
    contract.phase_id = PHASE_PREPROC;
    contract.x_work_base_word = x_base_word16;
    contract.w_base_word = w_base_word16;
    contract.token_range = make_token_range((u32_t)0u, (u32_t)N_NODES);
    contract.tile_range = make_tile_range((u32_t)0u, (u32_t)1u);

    fp16_rewrite::HeaderFp16PreprocWeightProvider weights;
    preproc_clean::PreprocFp16Debug dbg;
    u32_t input_words[CODE_N];
    PREPROC_WORD16_INPUT_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        input_words[v] = (topfed_in_payload != nullptr) ? topfed_in_payload[v] : (u32_t)0u;
    }
    preproc_clean::run_preproc_fp16_clean_from_words(input_words, weights, dbg);

    const uint32_t x_base = (uint32_t)x_base_word16.to_uint();
    PREPROC_WORD16_STORE_TOKEN_LOOP: for (uint32_t t = 0u; t < (uint32_t)N_NODES; ++t) {
        PREPROC_WORD16_STORE_DIM_LOOP: for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            const uint32_t idx = x_base + t * (uint32_t)D_MODEL + d;
            sram_word16[idx] = bits_from_fp16(dbg.x_work[t][d]);
        }
    }
}

} // namespace aecct
