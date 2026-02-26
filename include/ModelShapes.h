// ModelShapes.h
#pragma once
#include <cstdint>

// ============================================================
// ModelShapes.h
// ------------------------------------------------------------
// This header "freezes" the shapes for this specific model:
//   stage2_infer_frozen__BCH_n63_k51__Ndec2_d32_h8__...with_lpe_token.pth
//
// NOTE:
// - We use u32 word addressing everywhere (1 word = 4 bytes).
// - When commenting matrix/tensor shapes:
//     列(Row) = first dimension
//     行(Colume) = second dimension
// ============================================================

static const uint32_t BYTES_PER_WORD = 4;

// W_REGION wide-port lanes (each lane = 1 u32 word).
static const uint32_t W_LANES = 8;  // W_LANES=8 => 256-bit wide word

constexpr uint32_t ceil_div_u32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }
constexpr uint32_t align_up_u32(uint32_t x, uint32_t a) { return ((x + a - 1u) / a) * a; }

// FP32 payload in v8: TB sends fp32 as raw u32 words => 1 elem = 1 word.
constexpr uint32_t words_fp32(uint32_t elems) { return elems; }

// Bit-packed payload: 1 word carries 32 bits.
constexpr uint32_t words_bits(uint32_t bits) { return ceil_div_u32(bits, 32u); }

// ----------------------------
// BCH code parameters
// ----------------------------
static const uint32_t CODE_N  = 63;     // BCH n
static const uint32_t CODE_K  = 51;     // BCH k
static const uint32_t CODE_C  = 12;     // parity bits = n-k
static const uint32_t N_NODES = 75;    // nodes = CODE_N + CODE_C  (列(Row)=node)

// Parity-check matrix H shape: [CODE_C, CODE_N] (列(Row)=check, 行(Colume)=bit)
static const uint32_t H_ROWS = CODE_C;
static const uint32_t H_COLS = CODE_N;
static const uint32_t H_BITS = H_ROWS * H_COLS;
static const uint32_t H_WORDS_BITPACK = words_bits(H_BITS); // packed row-major

// ----------------------------
// Transformer parameters (from .pth)
// ----------------------------
static const uint32_t D_MODEL  = 32;   // (行(Colume)=d_model)
static const uint32_t N_HEAD   = 8;
static const uint32_t N_LAYERS = 2;
static const uint32_t D_HEAD   = (D_MODEL / N_HEAD);
static const uint32_t D_FFN    = 128;

static_assert((D_MODEL % N_HEAD) == 0, "D_MODEL must be divisible by N_HEAD");

// ----------------------------
// Extra feature dims (from .pth)
// ----------------------------
// src_embed: [N_NODES, D_SRC_EMBED] (列(Row)=node, 行(Colume)=embed_in)
static const uint32_t D_SRC_EMBED = 24;

// LPE tensor: [N_NODES, N_NODES, LPE_CH]
static const uint32_t LPE_CH = 2;

// lpe_proj: [D_LPE_PROJ, LPE_CH]
static const uint32_t D_LPE_PROJ = 8;

// lpe_token: [N_NODES, D_LPE_TOKEN]
static const uint32_t D_LPE_TOKEN = 8;

// ----------------------------
// Core runtime tensors (words)
// ----------------------------
// X: [N_NODES, D_MODEL]
static const uint32_t ELEMS_X = N_NODES * D_MODEL;
static const uint32_t WORDS_X_FP32 = words_fp32(ELEMS_X);

// y: [CODE_N]
static const uint32_t ELEMS_Y = CODE_N;
static const uint32_t WORDS_Y_FP32 = words_fp32(ELEMS_Y);

// src_mask: bool [1,1,N_NODES,N_NODES] in .pth
// For streaming/storage we treat it as [N_NODES,N_NODES] bitpack, row-major.
static const uint32_t SRC_MASK_BITS = N_NODES * N_NODES;
static const uint32_t SRC_MASK_WORDS_BITPACK = words_bits(SRC_MASK_BITS);

// Outputs (from .pth out_fc.weight shape => logits length = 63)
static const uint32_t ELEMS_X_PRED   = CODE_N;
static const uint32_t WORDS_X_PRED_FP32 = words_fp32(ELEMS_X_PRED);

static const uint32_t ELEMS_LOGITS   = 63;
static const uint32_t WORDS_LOGITS_FP32 = words_fp32(ELEMS_LOGITS);

// ----------------------------
// v8 expected_len (words)
// ----------------------------
// CFG_RX word count is defined by ModelDesc.h.

// INFER_RX payload = y
static const uint32_t EXP_LEN_INFER_IN_WORDS = WORDS_Y_FP32;

// OUT payload lengths
static const uint32_t EXP_LEN_OUT_XPRED_WORDS  = WORDS_X_PRED_FP32;
static const uint32_t EXP_LEN_OUT_LOGITS_WORDS = WORDS_LOGITS_FP32;

// LOAD_BIAS and LOAD_W lengths are defined by WeightStreamOrder.h,
// but we also expose legacy totals for quick sanity checks.
// Legacy (v11): EXP_LEN_BIAS_WORDS = 840
static const uint32_t EXP_LEN_BIAS_WORDS = 840;     // fp32 words (padded for W_LANES)
// Legacy (v11): EXP_LEN_W_WORDS = 43415
static const uint32_t EXP_LEN_W_WORDS    = 32328;   // includes padding for W_LANES alignment
// NOTE: Unified PARAM total length (EXP_LEN_PARAM_WORDS) is defined in WeightStreamOrder.h (v11.4 main path).
