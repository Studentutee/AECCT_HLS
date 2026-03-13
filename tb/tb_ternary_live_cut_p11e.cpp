// tb_ternary_live_cut_p11e.cpp
// P00-011E: strict minimal single-matrix live ternary cut proof for L0_WQ.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "AecctTypes.h"
#include "AecctProtocol.h"
#include "AecctUtil.h"
#include "QuantDesc.h"
#include "gen/SramMap.h"
#include "gen/WeightStreamOrder.h"
#include "Top.h"
#include "blocks/AttnLayer0.h"
#include "weights.h"

namespace {

struct L0WQExportRecord {
    uint32_t weight_param_id;
    uint32_t inv_sw_param_id;
    uint32_t rows;
    uint32_t cols;
    uint32_t num_weights;
    uint32_t payload_words_2b;
    uint32_t last_word_valid_count;
    uint32_t inv_sw_bits;
    std::vector<uint32_t> payload_words;
};

static void fail(const char* msg) {
    std::printf("[p11e][FAIL] %s\n", msg);
    std::exit(1);
}

static std::string read_text_file(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) {
        fail("cannot open gen/ternary_p11c_export.json");
    }
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static bool parse_u32_field(const std::string& text, const char* key, uint32_t& out) {
    const size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t pos = key_pos + std::strlen(key);
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
        ++pos;
    }
    char* end = 0;
    const unsigned long v = std::strtoul(text.c_str() + pos, &end, 10);
    if (end == text.c_str() + pos) {
        return false;
    }
    out = (uint32_t)v;
    return true;
}

static bool parse_hex_array_field(const std::string& text, const char* key, std::vector<uint32_t>& out_words) {
    const size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    const size_t array_lo = text.find('[', key_pos);
    const size_t array_hi = text.find(']', array_lo);
    if (array_lo == std::string::npos || array_hi == std::string::npos || array_hi <= array_lo) {
        return false;
    }

    size_t pos = array_lo;
    while (true) {
        const size_t hex_pos = text.find("0x", pos);
        if (hex_pos == std::string::npos || hex_pos >= array_hi) {
            break;
        }
        char* end = 0;
        const unsigned long v = std::strtoul(text.c_str() + hex_pos, &end, 16);
        if (end == text.c_str() + hex_pos) {
            return false;
        }
        out_words.push_back((uint32_t)v);
        pos = (size_t)(end - text.c_str());
    }
    return true;
}

static bool parse_l0_wq_export_record(const std::string& json_text, L0WQExportRecord& out) {
    const std::string needle = "\"matrix_id\": \"L0_WQ\"";
    const size_t matrix_pos = json_text.find(needle);
    if (matrix_pos == std::string::npos) {
        return false;
    }
    const size_t obj_start = json_text.rfind('{', matrix_pos);
    const size_t obj_end = json_text.find('}', matrix_pos);
    if (obj_start == std::string::npos || obj_end == std::string::npos || obj_end <= obj_start) {
        return false;
    }
    const std::string obj = json_text.substr(obj_start, obj_end - obj_start + 1u);

    std::vector<uint32_t> inv_sw_words;
    if (!parse_u32_field(obj, "\"weight_param_id\":", out.weight_param_id)) { return false; }
    if (!parse_u32_field(obj, "\"inv_sw_param_id\":", out.inv_sw_param_id)) { return false; }
    if (!parse_u32_field(obj, "\"rows\":", out.rows)) { return false; }
    if (!parse_u32_field(obj, "\"cols\":", out.cols)) { return false; }
    if (!parse_u32_field(obj, "\"num_weights\":", out.num_weights)) { return false; }
    if (!parse_u32_field(obj, "\"payload_words_2b\":", out.payload_words_2b)) { return false; }
    if (!parse_u32_field(obj, "\"last_word_valid_count\":", out.last_word_valid_count)) { return false; }
    if (!parse_hex_array_field(obj, "\"inv_sw_fp32_hex\":", inv_sw_words)) { return false; }
    if (!parse_hex_array_field(obj, "\"payload_hex_words\":", out.payload_words)) { return false; }
    if (inv_sw_words.empty()) {
        return false;
    }
    out.inv_sw_bits = inv_sw_words[0];
    return true;
}

static void expect_rsp(aecct::ctrl_ch_t& ctrl_rsp, uint8_t expect_kind, uint8_t expect_payload) {
    aecct::u16_t w;
    if (!ctrl_rsp.nb_read(w)) {
        fail("expected ctrl_rsp but channel empty");
    }
    const uint8_t kind = aecct::unpack_ctrl_rsp_kind(w);
    const uint8_t payload = aecct::unpack_ctrl_rsp_payload(w);
    if (kind != expect_kind || payload != expect_payload) {
        std::printf("[p11e][FAIL] rsp mismatch kind=%u payload=%u expect_kind=%u expect_payload=%u\n",
                    (unsigned)kind,
                    (unsigned)payload,
                    (unsigned)expect_kind,
                    (unsigned)expect_payload);
        std::exit(1);
    }
}

static void expect_state(aecct::TopState expect_s) {
    const aecct::TopState got = aecct::top_peek_state();
    if ((unsigned)got != (unsigned)expect_s) {
        std::printf("[p11e][FAIL] state mismatch got=%u expect=%u\n",
                    (unsigned)got,
                    (unsigned)expect_s);
        std::exit(1);
    }
}

static void tick(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out
) {
    aecct::top(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static void drive_cmd(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint8_t opcode
) {
    ctrl_cmd.write(aecct::pack_ctrl_cmd(opcode));
    tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
}

static void drive_set_w_base(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    uint32_t w_base_word
) {
    data_in.write((aecct::u32_t)w_base_word);
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SET_W_BASE);
}

static void send_param_words(
    aecct::ctrl_ch_t& ctrl_cmd,
    aecct::ctrl_ch_t& ctrl_rsp,
    aecct::data_ch_t& data_in,
    aecct::data_ch_t& data_out,
    const std::vector<uint32_t>& param_words
) {
    for (uint32_t i = 0u; i < (uint32_t)param_words.size(); ++i) {
        data_in.write((aecct::u32_t)param_words[i]);
        tick(ctrl_cmd, ctrl_rsp, data_in, data_out);
        if (i + 1u == (uint32_t)param_words.size()) {
            expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_LOAD_W);
        }
    }
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

static bool p11e_decode_ternary_code_lsb_2b(
    const aecct::u32_t* sram,
    uint32_t payload_base_word,
    uint32_t payload_words_2b,
    uint32_t elem_idx,
    uint32_t& out_code
) {
    const uint32_t word_idx = (elem_idx >> 4);
    if (word_idx >= payload_words_2b) {
        return false;
    }
    const uint32_t slot = (elem_idx & 15u);
    const uint32_t word = (uint32_t)sram[payload_base_word + word_idx].to_uint();
    out_code = (word >> (slot * 2u)) & 0x3u;
    return true;
}

static bool p11e_ternary_code_to_quant_weight(uint32_t code, aecct::quant_w_t& out_w) {
    if (code == 0u) {
        out_w = aecct::quant_w_t(0);
        return true;
    }
    if (code == 1u) {
        out_w = aecct::quant_w_t(1);
        return true;
    }
    if (code == 3u) {
        out_w = aecct::quant_w_t(-1);
        return true;
    }
    return false;
}

static bool p11e_attn_live_cut_l0_wq(
    aecct::u32_t* sram,
    const aecct::AttnCfg& cfg,
    aecct::u32_t x_in_base_word,
    const aecct::AttnScratch& sc,
    aecct::u32_t param_base_word
) {
    const QuantLinearMeta meta = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
    uint32_t token_count = (uint32_t)cfg.token_count.to_uint();
    uint32_t d_model = (uint32_t)cfg.d_model.to_uint();
    if (token_count == 0u) { token_count = 1u; }
    if (d_model == 0u) { d_model = meta.cols; }
    if (meta.rows == 0u || meta.cols == 0u) {
        return false;
    }
    if (d_model != meta.cols) {
        return false;
    }

    const uint32_t param_base = (uint32_t)param_base_word.to_uint();
    const uint32_t x_in_base = (uint32_t)x_in_base_word.to_uint();
    const uint32_t q_base = (uint32_t)sc.q_base_word.to_uint();
    const uint32_t q_sx_base = (uint32_t)sc.q_sx_base_word.to_uint();
    const uint32_t payload_base = param_base + kParamMeta[meta.weight_param_id].offset_w;
    const uint32_t inv_sw_base = param_base + kParamMeta[meta.inv_sw_param_id].offset_w;

    const aecct::u32_t inv_sw_bits = sram[inv_sw_base];
    aecct::fp32_t inv_sw_fp = aecct::fp32_from_bits(inv_sw_bits);
    aecct::quant_acc_t inv_sw = inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    if (inv_sw == aecct::quant_acc_t(0)) {
        return false;
    }

    sram[q_sx_base] = inv_sw_bits;

    for (uint32_t t = 0; t < token_count; ++t) {
        const uint32_t x_row = x_in_base + t * meta.cols;
        const uint32_t q_row = q_base + t * meta.rows;
        for (uint32_t out = 0; out < meta.rows; ++out) {
            aecct::quant_acc_t acc = 0;
            const uint32_t w_row = out * meta.cols;
            for (uint32_t in = 0; in < meta.cols; ++in) {
                uint32_t code = 0u;
                if (!p11e_decode_ternary_code_lsb_2b(
                        sram,
                        payload_base,
                        meta.payload_words_2b,
                        w_row + in,
                        code)) {
                    return false;
                }
                aecct::quant_w_t w = 0;
                if (!p11e_ternary_code_to_quant_weight(code, w)) {
                    return false;
                }
                aecct::quant_act_t x = aecct::quant_act_from_bits(sram[x_row + in]);
                acc += aecct::quant_acc_t(x) * aecct::quant_acc_t(w);
            }
            sram[q_row + out] = aecct::quant_bits_from_acc(acc / inv_sw);
        }
    }

    return true;
}

static void compare_exact_or_die(const char* label, uint32_t got_bits, uint32_t expect_bits, uint32_t idx) {
    if (got_bits != expect_bits) {
        std::printf("[p11e][FAIL] %s mismatch idx=%u got=0x%08X expect=0x%08X got_f=%g expect_f=%g\n",
                    label,
                    (unsigned)idx,
                    (unsigned)got_bits,
                    (unsigned)expect_bits,
                    (double)bits_to_f32(got_bits),
                    (double)bits_to_f32(expect_bits));
        std::exit(1);
    }
}

} // namespace

int main() {
    const std::string json_text = read_text_file("gen/ternary_p11c_export.json");
    L0WQExportRecord rec;
    if (!parse_l0_wq_export_record(json_text, rec)) {
        fail("failed to parse L0_WQ record from gen/ternary_p11c_export.json");
    }

    const QuantLinearMeta meta = kQuantLinearMeta[(uint32_t)QLM_L0_WQ];
    if (rec.weight_param_id != meta.weight_param_id ||
        rec.inv_sw_param_id != meta.inv_sw_param_id ||
        rec.rows != meta.rows ||
        rec.cols != meta.cols ||
        rec.num_weights != meta.num_weights ||
        rec.payload_words_2b != meta.payload_words_2b ||
        rec.last_word_valid_count != meta.last_word_valid_count) {
        fail("L0_WQ JSON record does not match current metadata");
    }
    if (rec.payload_words.size() != rec.payload_words_2b) {
        fail("payload_words size mismatch");
    }

    const uint32_t expect_inv_sw_bits = (uint32_t)aecct::fp32_bits_from_double(1.0 / w_decoder_layers_0_self_attn_linears_0_s_w[0]).to_uint();
    if (rec.inv_sw_bits != expect_inv_sw_bits) {
        fail("inv_s_w JSON record does not match weights.h reference");
    }

    std::vector<uint32_t> param_words((uint32_t)EXP_LEN_PARAM_WORDS, 0u);
    const ParamMeta payload_meta = kParamMeta[rec.weight_param_id];
    const ParamMeta inv_meta = kParamMeta[rec.inv_sw_param_id];
    for (uint32_t i = 0u; i < rec.payload_words_2b; ++i) {
        param_words[payload_meta.offset_w + i] = rec.payload_words[i];
    }
    param_words[inv_meta.offset_w] = rec.inv_sw_bits;

    aecct::ctrl_ch_t ctrl_cmd;
    aecct::ctrl_ch_t ctrl_rsp;
    aecct::data_ch_t data_in;
    aecct::data_ch_t data_out;

    const uint32_t param_base_word = (uint32_t)sram_map::PARAM_BASE_DEFAULT;
    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_SOFT_RESET);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_DONE, (uint8_t)aecct::OP_SOFT_RESET);
    expect_state(aecct::ST_IDLE);

    drive_set_w_base(ctrl_cmd, ctrl_rsp, data_in, data_out, param_base_word);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_SET_W_BASE);
    expect_state(aecct::ST_IDLE);

    drive_cmd(ctrl_cmd, ctrl_rsp, data_in, data_out, (uint8_t)aecct::OP_LOAD_W);
    expect_rsp(ctrl_rsp, (uint8_t)aecct::RSP_OK, (uint8_t)aecct::OP_LOAD_W);
    expect_state(aecct::ST_PARAM_RX);
    send_param_words(ctrl_cmd, ctrl_rsp, data_in, data_out, param_words);
    expect_state(aecct::ST_IDLE);

    aecct::u32_t* sram = aecct::top_sram();
    const uint32_t payload_base = param_base_word + payload_meta.offset_w;
    const uint32_t inv_base = param_base_word + inv_meta.offset_w;
    compare_exact_or_die("live payload first word", (uint32_t)sram[payload_base + 0u].to_uint(), rec.payload_words[0u], 0u);
    compare_exact_or_die("live payload last word", (uint32_t)sram[payload_base + rec.payload_words_2b - 1u].to_uint(), rec.payload_words[rec.payload_words_2b - 1u], rec.payload_words_2b - 1u);
    compare_exact_or_die("live inv_s_w word", (uint32_t)sram[inv_base].to_uint(), rec.inv_sw_bits, 0u);

    const uint32_t x_in_base = (uint32_t)aecct::ATTN_X_IN_BASE_WORD_DEFAULT;
    for (uint32_t i = 0u; i < rec.cols; ++i) {
        const float x = (float)((int)(i % 7u) - 3) * 0.25f;
        sram[x_in_base + i] = (aecct::u32_t)f32_to_bits(x);
    }

    aecct::AttnCfg cfg;
    cfg.token_count = (aecct::u32_t)1u;
    cfg.d_model = (aecct::u32_t)rec.cols;
    cfg.n_heads = (aecct::u32_t)1u;
    cfg.d_head = (aecct::u32_t)rec.cols;

    aecct::AttnScratch sc;
    sc.q_base_word = (aecct::u32_t)sram_map::BASE_SCR_K_W;
    sc.k_base_word = 0;
    sc.v_base_word = 0;
    sc.score_base_word = 0;
    sc.softmax_base_word = 0;
    sc.pre_concat_base_word = 0;
    sc.post_concat_base_word = 0;
    sc.q_act_q_base_word = 0;
    sc.k_act_q_base_word = 0;
    sc.v_act_q_base_word = 0;
    sc.q_sx_base_word = (aecct::u32_t)(sram_map::BASE_SCR_K_W + rec.rows);

    if (!p11e_attn_live_cut_l0_wq(
            sram,
            cfg,
            (aecct::u32_t)x_in_base,
            sc,
            (aecct::u32_t)param_base_word)) {
        fail("p11e_attn_live_cut_l0_wq returned false");
    }

    compare_exact_or_die("q_sx/inv_s_w binding", (uint32_t)sram[(uint32_t)sc.q_sx_base_word.to_uint()].to_uint(), rec.inv_sw_bits, 0u);

    aecct::fp32_t inv_sw_fp = aecct::fp32_from_bits((aecct::u32_t)rec.inv_sw_bits);
    aecct::quant_acc_t inv_sw = inv_sw_fp.template convert_to_ac_fixed<32, 12, true, AC_RND, AC_SAT>(false);
    for (uint32_t out = 0u; out < rec.rows; ++out) {
        aecct::quant_acc_t acc = 0;
        const uint32_t w_row = out * rec.cols;
        for (uint32_t in = 0u; in < rec.cols; ++in) {
            const uint32_t x_bits = (uint32_t)sram[x_in_base + in].to_uint();
            const aecct::quant_act_t x = aecct::quant_act_from_bits((aecct::u32_t)x_bits);
            const aecct::quant_w_t w = aecct::quant_w_t((int)w_decoder_layers_0_self_attn_linears_0_weight[w_row + in]);
            acc += aecct::quant_acc_t(x) * aecct::quant_acc_t(w);
        }
        const uint32_t expect_bits = (uint32_t)aecct::quant_bits_from_acc(acc / inv_sw).to_uint();
        const uint32_t got_bits = (uint32_t)sram[(uint32_t)sc.q_base_word.to_uint() + out].to_uint();
        compare_exact_or_die("L0_WQ live q", got_bits, expect_bits, out);
    }

    std::printf("[p11e][PASS] matrix=L0_WQ live load words=%u inv_sw_bits=0x%08X\n",
                (unsigned)rec.payload_words_2b,
                (unsigned)rec.inv_sw_bits);
    std::printf("[p11e][PASS] matrix=L0_WQ live consumer decode/binding exact-bit match rows=%u cols=%u\n",
                (unsigned)rec.rows,
                (unsigned)rec.cols);
    std::printf("PASS: tb_ternary_live_cut_p11e\n");
    return 0;
}
