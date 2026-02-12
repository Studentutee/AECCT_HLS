#pragma once

#include "ac_channel.h"
#include "ac_std_float.h"
#include "ac_int.h"

// ============================================================
// PreprocEmbedSPE (external weight memory version)
// - weights live in TOP SRAM
// - this block reads via base offsets
// - runtime-configurable sizes but bounded by MAX_*
// ============================================================
class PreprocEmbedSPE_ext {
public:
  // ----------------------------
  // Compile-time maxima (edit as needed)
  // ----------------------------

  static constexpr int MAX_CODE_N  = PREPROC_MAX_CODE_N;
  static constexpr int MAX_CODE_C  = PREPROC_MAX_CODE_C;
  static constexpr int MAX_N_NODES = PREPROC_MAX_CODE_N + PREPROC_MAX_CODE_C;
  static constexpr int MAX_D_EMBED = PREPROC_MAX_D_EMBED;
  static constexpr int MAX_D_SPE   = PREPROC_MAX_D_SPE;
  static constexpr int MAX_D_MODEL = PREPROC_MAX_D_EMBED + PREPROC_MAX_D_SPE;

  // ----------------------------
  // Runtime config container
  // ----------------------------
  struct Cfg {
    ac_int<16, false> code_n;   // 碼字長度 N
    ac_int<16, false> code_c;   // 校驗數 C
    ac_int<16, false> d_embed;  // embed dim
    ac_int<16, false> d_spe;    // SPE token dim
  };

private:
  Cfg cfg_reg;

  // Base offsets (in TOP SRAM address space)
  ac_int<32, false> base_src_embed;
  ac_int<32, false> base_lpe_token;
  ac_int<32, false> base_h_bits;

public:
  PreprocEmbedSPE_ext() {
    cfg_reg.code_n  = 0;
    cfg_reg.code_c  = 0;
    cfg_reg.d_embed = 0;
    cfg_reg.d_spe   = 0;
    base_src_embed  = 0;
    base_lpe_token  = 0;
    base_h_bits     = 0;
  }

  void configure(const Cfg &c,
                 const ac_int<32, false> b_src,
                 const ac_int<32, false> b_lpe,
                 const ac_int<32, false> b_h) {
    cfg_reg = c;
    base_src_embed = b_src;
    base_lpe_token = b_lpe;
    base_h_bits    = b_h;
  }

  // TOP 提供的 SRAM read 介面（用函式指標/ functor 最通用）
  // - read_f32(addr) : returns ac_ieee_float<binary32>
  // - read_b1(addr)  : returns ac_int<1,false>
  template <typename MemReadF32, typename MemReadB1>
  void run(ac_channel<ac_ieee_float<binary32>> &y_in_ch,
           ac_channel<ac_ieee_float<binary32>> &out_ch,
           MemReadF32 read_f32,
           MemReadB1  read_b1) {

    const int code_n  = (int)cfg_reg.code_n;
    const int code_c  = (int)cfg_reg.code_c;
    const int n_nodes = code_n + code_c;
    const int d_embed = (int)cfg_reg.d_embed;
    const int d_spe   = (int)cfg_reg.d_spe;
    const int d_model = d_embed + d_spe;

    // Hard-bit pack of y (1 => negative)
    ac_int<1, false> hb_pack[MAX_CODE_N];

    // ----------------------------
    // Nodes [0 .. CODE_N-1]: abs(y) * embed + SPE token
    // ----------------------------
    NODE_Y_LOOP: for (int node_idx = 0; node_idx < code_n; ++node_idx) {
      ac_ieee_float<binary32> y_val = y_in_ch.read();
      hb_pack[node_idx] = (y_val < ac_ieee_float<binary32>(0.0f)) ? ac_int<1,false>(1) : ac_int<1,false>(0);

      ac_ieee_float<binary32> abs_y = (y_val < ac_ieee_float<binary32>(0.0f)) ? (-y_val) : y_val;

      FEAT_Y_LOOP: for (int feat_idx = 0; feat_idx < d_model; ++feat_idx) {
        ac_ieee_float<binary32> out_val;
        if (feat_idx < d_embed) {
          const ac_int<32,false> w_addr = base_src_embed + (ac_int<32,false>)(node_idx * d_embed + feat_idx);
          const ac_ieee_float<binary32> w = read_f32(w_addr);
          out_val = abs_y * w;
        } else {
          const int spe_idx = feat_idx - d_embed;
          const ac_int<32,false> w_addr = base_lpe_token + (ac_int<32,false>)(node_idx * d_spe + spe_idx);
          const ac_ieee_float<binary32> w = read_f32(w_addr);
          out_val = w;
        }
        out_ch.write(out_val);
      }
    }

    // ----------------------------
    // Nodes [CODE_N .. N_NODES-1]: syndrome (+1/-1) * embed + SPE token
    // ----------------------------
    NODE_SYN_LOOP: for (int chk_idx = 0; chk_idx < code_c; ++chk_idx) {
      ac_int<1,false> parity = 0;

      SYN_COL_LOOP: for (int col_idx = 0; col_idx < code_n; ++col_idx) {
        const ac_int<32,false> h_addr = base_h_bits + (ac_int<32,false>)(chk_idx * code_n + col_idx);
        const ac_int<1,false> h_bit = read_b1(h_addr);
        parity ^= (h_bit & hb_pack[col_idx]);
      }

      ac_ieee_float<binary32> syn_val =
        (parity == 0) ? ac_ieee_float<binary32>(1.0f) : ac_ieee_float<binary32>(-1.0f);

      const int node_idx = code_n + chk_idx;

      FEAT_SYN_LOOP: for (int feat_idx = 0; feat_idx < d_model; ++feat_idx) {
        ac_ieee_float<binary32> out_val;
        if (feat_idx < d_embed) {
          const ac_int<32,false> w_addr = base_src_embed + (ac_int<32,false>)(node_idx * d_embed + feat_idx);
          const ac_ieee_float<binary32> w = read_f32(w_addr);
          out_val = syn_val * w;
        } else {
          const int spe_idx = feat_idx - d_embed;
          const ac_int<32,false> w_addr = base_lpe_token + (ac_int<32,false>)(node_idx * d_spe + spe_idx);
          const ac_ieee_float<binary32> w = read_f32(w_addr);
          out_val = w;
        }
        out_ch.write(out_val);
      }
    }
  }
};
