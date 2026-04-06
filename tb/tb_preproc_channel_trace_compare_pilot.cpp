#ifndef __SYNTHESIS__

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "AecctTypes.h"
#include "blocks/PreprocEmbedSPEChannelTop.h"
#include "blocks/PreprocTransportTypes.h"
#include "embed_plus_SPE_step0.h"
#include "input_y_step0.h"
#include "weights.h"

namespace {

struct TraceCompareStats {
    uint32_t mismatch_count;
    uint32_t first_token;
    uint32_t first_dim;
    uint32_t first_got_bits;
    uint32_t first_ref_bits;
    float max_abs_diff;
};

static uint32_t f32_to_bits(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_to_f32(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static void clear_trace_compare_stats(TraceCompareStats& s) {
    s.mismatch_count = 0u;
    s.first_token = 0u;
    s.first_dim = 0u;
    s.first_got_bits = 0u;
    s.first_ref_bits = 0u;
    s.max_abs_diff = 0.0f;
}

static void clear_u32_words(aecct::u32_t* dst, uint32_t words) {
    CLEAR_U32_WORDS_LOOP: for (uint32_t i = 0u; i < words; ++i) {
        dst[i] = (aecct::u32_t)0u;
    }
}

static void build_embed_lpe_token_words(
    aecct::u32_t* embed_words_token_major,
    aecct::u32_t* lpe_words_token_major
) {
    BUILD_EMBED_TOKEN_LOOP: for (uint32_t token = 0u; token < (uint32_t)N_NODES; ++token) {
        BUILD_EMBED_WORD_LOOP: for (uint32_t i = 0u; i < (uint32_t)D_SRC_EMBED; ++i) {
            const double src = w_src_embed[token * (uint32_t)D_SRC_EMBED + i];
            embed_words_token_major[token * (uint32_t)D_SRC_EMBED + i] =
                (aecct::u32_t)f32_to_bits((float)src);
        }
        BUILD_LPE_WORD_LOOP: for (uint32_t i = 0u; i < (uint32_t)D_LPE_TOKEN; ++i) {
            const double src = w_lpe_token[token * (uint32_t)D_LPE_TOKEN + i];
            lpe_words_token_major[token * (uint32_t)D_LPE_TOKEN + i] =
                (aecct::u32_t)f32_to_bits((float)src);
        }
    }
}

static void build_input_y_words(uint32_t sample_idx, aecct::u32_t* y_words_var_major) {
    BUILD_INPUT_Y_LOOP: for (uint32_t v = 0u; v < (uint32_t)CODE_N; ++v) {
        const double src = trace_input_y_step0_tensor[sample_idx * (uint32_t)CODE_N + v];
        y_words_var_major[v] = (aecct::u32_t)f32_to_bits((float)src);
    }
}

static uint32_t count_h_adj_for_var(uint32_t var_idx) {
    uint32_t count = 0u;
    COUNT_H_ADJ_LOOP: for (uint32_t c = 0u; c < (uint32_t)CODE_C; ++c) {
        const uint32_t flat = c * (uint32_t)CODE_N + var_idx;
        if ((uint32_t)h_H[flat].to_uint() != 0u) {
            ++count;
        }
    }
    return count;
}

static int run_one_sample(
    uint32_t sample_idx,
    const aecct::u32_t* embed_words_token_major,
    const aecct::u32_t* lpe_words_token_major,
    TraceCompareStats& out_stats
) {
    clear_trace_compare_stats(out_stats);

    aecct::preproc_y_in_ch_t y_in_ch;
    aecct::preproc_h_by_var_adj_ch_t h_by_var_adj_ch;
    aecct::preproc_embed_param_ch_t embed_param_ch;
    aecct::preproc_lpe_token_ch_t lpe_token_ch;
    aecct::preproc_check_acc_rd_ch_t check_acc_rd_ch;
    aecct::preproc_check_acc_wr_ch_t check_acc_wr_ch;
    aecct::preproc_x_out_ch_t preproc_x_out_ch;

    aecct::u32_t check_backing_words[
        aecct::PREPROC_PILOT_CHECK_TILE_COUNT * aecct::PREPROC_PILOT_CHECK_PARITY_TILE_WORDS];
    aecct::u32_t x_out_words_token_major[(uint32_t)N_NODES * (uint32_t)D_MODEL];
    aecct::u32_t input_y_words[(uint32_t)CODE_N];

    clear_u32_words(
        check_backing_words,
        (uint32_t)(aecct::PREPROC_PILOT_CHECK_TILE_COUNT * aecct::PREPROC_PILOT_CHECK_PARITY_TILE_WORDS));
    clear_u32_words(x_out_words_token_major, (uint32_t)N_NODES * (uint32_t)D_MODEL);
    build_input_y_words(sample_idx, input_y_words);

    if (sample_idx == 0u) {
        const uint32_t y_bits = (uint32_t)input_y_words[0].to_uint();
        const float y_float = bits_to_f32(y_bits);
        const uint32_t adj_count = count_h_adj_for_var(0u);
        std::printf(
            "PREPROC_DEBUG_FIRST_VAR_PACKET sample=0 var_idx=0 y_bits=0x%08X y_float=%.9g adj_count=%u embed_token_kind=%u embed_token_idx=%u lpe_token_kind=%u lpe_token_idx=%u\n",
            (unsigned)y_bits,
            (double)y_float,
            (unsigned)adj_count,
            (unsigned)aecct::PREPROC_PILOT_TOKEN_VAR,
            0u,
            (unsigned)aecct::PREPROC_PILOT_TOKEN_VAR,
            0u);
        PREPROC_DEBUG_FIRST_VAR_DIM_LOOP: for (uint32_t d = 0u; d < 8u; ++d) {
            const uint32_t embed_bits = (uint32_t)embed_words_token_major[d].to_uint();
            const uint32_t lpe_bits = (uint32_t)lpe_words_token_major[d].to_uint();
            std::printf(
                "PREPROC_DEBUG_FIRST_VAR_PACKET_DIM d=%u embed=%.9g(0x%08X) lpe=%.9g(0x%08X)\n",
                (unsigned)d,
                (double)bits_to_f32(embed_bits),
                (unsigned)embed_bits,
                (double)bits_to_f32(lpe_bits),
                (unsigned)lpe_bits);
        }
    }

    aecct::preproc_top_seed_check_acc_rd_from_backing(check_backing_words, check_acc_rd_ch);
    aecct::preproc_top_emit_y_var_major(input_y_words, y_in_ch);
    aecct::preproc_top_emit_h_by_var_adj_from_h_matrix(h_H, h_by_var_adj_ch);
    aecct::preproc_top_emit_embed_lpe_payloads(
        embed_words_token_major,
        lpe_words_token_major,
        embed_param_ch,
        lpe_token_ch);

    aecct::PreprocChannelPilotStats pilot_stats;
    aecct::preproc_channel_pilot_stats_clear(pilot_stats);
    const bool ok = aecct::preproc_embed_spe_channel_top(
        y_in_ch,
        h_by_var_adj_ch,
        embed_param_ch,
        lpe_token_ch,
        check_acc_rd_ch,
        check_acc_wr_ch,
        preproc_x_out_ch,
        &pilot_stats);
    if (!ok || pilot_stats.metadata_error) {
        std::printf(
            "TRACE_CHANNEL_RUN_ERROR sample=%u metadata_error=%u\n",
            (unsigned)sample_idx,
            pilot_stats.metadata_error ? 1u : 0u);
        return 1;
    }

    const uint32_t check_writes =
        aecct::preproc_top_drain_check_acc_wr_to_backing(check_acc_wr_ch, check_backing_words);
    uint32_t out_tokens = 0u;
    bool first_out_valid = false;
    aecct::PreprocXOutPacket first_out_pkt;
    aecct::preproc_x_out_packet_clear(first_out_pkt);
    aecct::PreprocXOutPacket out_pkt;
    PREPROC_DRAIN_X_OUT_LOOP: while (preproc_x_out_ch.nb_read(out_pkt)) {
        if (!first_out_valid) {
            first_out_pkt = out_pkt;
            first_out_valid = true;
        }
        const uint32_t token_idx = (uint32_t)out_pkt.token_idx.to_uint();
        const uint32_t token_kind = (uint32_t)out_pkt.token_kind.to_uint();
        uint32_t token_order = token_idx;
        if (token_kind == (uint32_t)aecct::PREPROC_PILOT_TOKEN_CHECK) {
            token_order = (uint32_t)aecct::PREPROC_PILOT_VAR_TOKENS + token_idx;
        }
        if (token_order >= (uint32_t)N_NODES) {
            continue;
        }
        const uint32_t words = (uint32_t)out_pkt.word_count.to_uint();
        const uint32_t clipped =
            (words > (uint32_t)aecct::PREPROC_PILOT_X_WORDS) ? (uint32_t)aecct::PREPROC_PILOT_X_WORDS : words;
        PREPROC_DRAIN_X_OUT_WORD_LOOP: for (uint32_t d = 0u; d < clipped; ++d) {
            x_out_words_token_major[token_order * (uint32_t)D_MODEL + d] = out_pkt.x_words[d];
        }
        ++out_tokens;
    }
    if (out_tokens != (uint32_t)N_NODES) {
        std::printf(
            "TRACE_CHANNEL_TOKEN_COUNT_MISMATCH sample=%u out_tokens=%u expect=%u\n",
            (unsigned)sample_idx,
            (unsigned)out_tokens,
            (unsigned)N_NODES);
        return 1;
    }
    if (!first_out_valid) {
        std::printf("TRACE_CHANNEL_FIRST_OUT_MISSING sample=%u\n", (unsigned)sample_idx);
        return 1;
    }

    if (sample_idx == 0u) {
        const uint32_t out_kind = (uint32_t)first_out_pkt.token_kind.to_uint();
        const uint32_t out_idx = (uint32_t)first_out_pkt.token_idx.to_uint();
        const uint32_t out_words = (uint32_t)first_out_pkt.word_count.to_uint();
        std::printf(
            "PREPROC_DEBUG_FIRST_OUT_PACKET sample=0 token_kind=%u token_idx=%u word_count=%u\n",
            (unsigned)out_kind,
            (unsigned)out_idx,
            (unsigned)out_words);
        PREPROC_DEBUG_FIRST_OUT_COMPARE_LOOP: for (uint32_t d = 0u; d < 8u; ++d) {
            const uint32_t got_bits = (uint32_t)first_out_pkt.x_words[d].to_uint();
            const float ref_f = (float)trace_embed_plus_SPE_step0_tensor[d];
            const uint32_t ref_bits = f32_to_bits(ref_f);
            std::printf(
                "PREPROC_DEBUG_FIRST_OUT_COMPARE d=%u got=%.9g(0x%08X) ref=%.9g(0x%08X)\n",
                (unsigned)d,
                (double)bits_to_f32(got_bits),
                (unsigned)got_bits,
                (double)ref_f,
                (unsigned)ref_bits);
        }
    }

    const size_t sample_base = (size_t)sample_idx * (size_t)N_NODES * (size_t)D_MODEL;
    COMPARE_TOKEN_LOOP: for (uint32_t token = 0u; token < (uint32_t)N_NODES; ++token) {
        COMPARE_DIM_LOOP: for (uint32_t d = 0u; d < (uint32_t)D_MODEL; ++d) {
            const uint32_t got_bits =
                (uint32_t)x_out_words_token_major[token * (uint32_t)D_MODEL + d].to_uint();
            const float ref_f =
                (float)trace_embed_plus_SPE_step0_tensor[sample_base + token * (size_t)D_MODEL + d];
            const uint32_t ref_bits = f32_to_bits(ref_f);

            const float got_f = bits_to_f32(got_bits);
            const float abs_diff = (float)std::fabs((double)got_f - (double)ref_f);
            if (abs_diff > out_stats.max_abs_diff) {
                out_stats.max_abs_diff = abs_diff;
            }

            if (got_bits != ref_bits) {
                if (out_stats.mismatch_count == 0u) {
                    out_stats.first_token = token;
                    out_stats.first_dim = d;
                    out_stats.first_got_bits = got_bits;
                    out_stats.first_ref_bits = ref_bits;
                }
                out_stats.mismatch_count = out_stats.mismatch_count + 1u;
            }
        }
    }

    std::printf(
        "TRACE_CHANNEL_STATS sample=%u var_tokens=%u check_tokens=%u check_tiles_loaded=%u check_tiles_written=%u check_writes=%u out_tokens=%u\n",
        (unsigned)sample_idx,
        (unsigned)pilot_stats.var_tokens_consumed.to_uint(),
        (unsigned)pilot_stats.check_tokens_emitted.to_uint(),
        (unsigned)pilot_stats.check_tiles_loaded.to_uint(),
        (unsigned)pilot_stats.check_tiles_written.to_uint(),
        (unsigned)check_writes,
        (unsigned)out_tokens);

    std::printf(
        "TRACE_COMPARE_RESULT sample=%u mode=exact mismatch_count=%u first_token=%u first_dim=%u first_got=0x%08X first_ref=0x%08X max_abs_diff=%.9g\n",
        (unsigned)sample_idx,
        (unsigned)out_stats.mismatch_count,
        (unsigned)out_stats.first_token,
        (unsigned)out_stats.first_dim,
        (unsigned)out_stats.first_got_bits,
        (unsigned)out_stats.first_ref_bits,
        (double)out_stats.max_abs_diff);

    return 0;
}

} // namespace

int main() {
    if (trace_input_y_step0_tensor_shape[0] != trace_embed_plus_SPE_step0_tensor_shape[0]) {
        std::printf("TRACE_SHAPE_MISMATCH batch input=%d embed_plus=%d\n",
            trace_input_y_step0_tensor_shape[0],
            trace_embed_plus_SPE_step0_tensor_shape[0]);
        return 1;
    }
    if (trace_input_y_step0_tensor_shape[1] != (int)CODE_N) {
        std::printf("TRACE_SHAPE_MISMATCH input_y cols=%d expect=%u\n",
            trace_input_y_step0_tensor_shape[1],
            (unsigned)CODE_N);
        return 1;
    }
    if (trace_embed_plus_SPE_step0_tensor_shape[1] != (int)N_NODES ||
        trace_embed_plus_SPE_step0_tensor_shape[2] != (int)D_MODEL) {
        std::printf("TRACE_SHAPE_MISMATCH embed_plus shape=(%d,%d) expect=(%u,%u)\n",
            trace_embed_plus_SPE_step0_tensor_shape[1],
            trace_embed_plus_SPE_step0_tensor_shape[2],
            (unsigned)N_NODES,
            (unsigned)D_MODEL);
        return 1;
    }

    static aecct::u32_t embed_words_token_major[(uint32_t)N_NODES * (uint32_t)D_SRC_EMBED];
    static aecct::u32_t lpe_words_token_major[(uint32_t)N_NODES * (uint32_t)D_LPE_TOKEN];
    build_embed_lpe_token_words(embed_words_token_major, lpe_words_token_major);

    const uint32_t sample_indices[3] = {0u, 1u, 2u};
    uint32_t total_mismatch = 0u;
    float global_max_abs_diff = 0.0f;

    std::printf(
        "TRACE_COMPARE_TARGET tensor=trace_embed_plus_SPE_step0_tensor file=data/trace/embed_plus_SPE_step0.h\n");
    std::printf("TRACE_COMPARE_RULE mode=exact atol=0 rtol=0\n");

    SAMPLE_LOOP: for (uint32_t i = 0u; i < 3u; ++i) {
        TraceCompareStats sample_stats;
        clear_trace_compare_stats(sample_stats);
        if (run_one_sample(sample_indices[i], embed_words_token_major, lpe_words_token_major, sample_stats) != 0) {
            return 1;
        }
        total_mismatch = total_mismatch + sample_stats.mismatch_count;
        if (sample_stats.max_abs_diff > global_max_abs_diff) {
            global_max_abs_diff = sample_stats.max_abs_diff;
        }
    }

    std::printf(
        "TRACE_COMPARE_AGGREGATE samples=3 mismatch_count=%u max_abs_diff=%.9g\n",
        (unsigned)total_mismatch,
        (double)global_max_abs_diff);

    if (total_mismatch != 0u) {
        std::printf("PREPROC_TRACE_COMPARE FAIL\n");
        return 1;
    }

    std::printf("PREPROC_CHANNEL_TRANSPORT_SKELETON PASS\n");
    std::printf("PREPROC_TRACE_COMPARE PASS\n");
    std::printf("PASS: tb_preproc_channel_trace_compare_pilot\n");
    return 0;
}

#endif // __SYNTHESIS__
