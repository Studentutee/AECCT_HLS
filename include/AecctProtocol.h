#pragma once
// AecctProtocol.h
// ctrl_cmd / ctrl_rsp 編碼與 opcode / error code 定義（M0 釘死）
//
// ctrl_cmd  (16-bit): [7:0]=opcode, [15:8]=0
// ctrl_rsp  (16-bit): [3:0]=rsp_kind(0 OK,1 DONE,2 ERR), [7:4]=0, [15:8]=payload
//   payload: OK/DONE -> opcode, ERR -> err_code
//
// 注意：M0 先把 enum 列完整；未實作的 opcode 先回 ERR_UNIMPL

#include <ac_int.h>

namespace aecct {

	// -------------------- RSP kind --------------------
	enum RspKind : unsigned {
		RSP_OK = 0,
		RSP_DONE = 1,
		RSP_ERR = 2
	};

	// -------------------- Opcode --------------------
	enum Opcode : unsigned {
		OP_NOOP = 0x00, // M0: implemented
		OP_CFG_BEGIN = 0x01,
		OP_CFG_COMMIT = 0x02,

		OP_LOAD_W = 0x04, // LOAD_PARAM path -> M1+
		OP_SET_OUTMODE = 0x05,
		OP_INFER = 0x06,
		OP_READ_MEM = 0x07,
		OP_DEBUG_CFG = 0x08,
		OP_SET_W_BASE = 0x09,

		OP_SOFT_RESET = 0x0F  // M0: implemented
	};

	// -------------------- Top FSM state (M1) --------------------
	enum TopState : unsigned {
		ST_IDLE = 0,     // 只接受控制命令 (control command only)
		ST_CFG_RX,       // 組態接收階段 (cfg receive phase)
		ST_PARAM_RX,     // 參數接收階段 (param/weight receive phase)
		ST_INFER_RX,     // 推論輸入接收階段 (infer input receive phase)
		ST_HALTED        // 停機占位狀態 (debug halt placeholder)
	};

	// -------------------- Error code --------------------
	enum ErrCode : unsigned {
		ERR_OK = 0,
		ERR_UNIMPL,
		ERR_BAD_STATE,
		ERR_BAD_ARG,
		ERR_BUSY,
		ERR_INTERNAL,

		// placeholders for M3/M4+
		ERR_CFG_LEN_MISMATCH,
		ERR_CFG_ILLEGAL,
		ERR_PARAM_LEN_MISMATCH,
		ERR_PARAM_BASE_RANGE,
		ERR_PARAM_BASE_ALIGN,
		ERR_MEM_RANGE,
		ERR_BITPACK_PAD,
		ERR_DBG_HALT
	};

	// -------------------- ctrl word pack/unpack helpers --------------------
	typedef ac_int<16, false> ctrl_word_t;

	static inline ctrl_word_t pack_ctrl_cmd(uint8_t opcode_u8) {
		// [7:0]=opcode, [15:8]=0
		ctrl_word_t w = 0;
		w.set_slc(0, (ac_int<8, false>)opcode_u8);
		return w;
	}

	static inline uint8_t unpack_ctrl_cmd_opcode(const ctrl_word_t& w) {
		ac_int<8, false> op = w.template slc<8>(0);
		return (uint8_t)op.to_uint();
	}

	static inline ctrl_word_t pack_ctrl_rsp_done(uint8_t opcode_u8) {
		// rsp_kind=DONE, payload=opcode
		ctrl_word_t w = 0;
		w.set_slc(0, (ac_int<4, false>)RSP_DONE);         // [3:0]
		w.set_slc(8, (ac_int<8, false>)opcode_u8);        // [15:8]
		return w;
	}

	static inline ctrl_word_t pack_ctrl_rsp_ok(uint8_t opcode_u8) {
		ctrl_word_t w = 0;
		w.set_slc(0, (ac_int<4, false>)RSP_OK);
		w.set_slc(8, (ac_int<8, false>)opcode_u8);
		return w;
	}

	static inline ctrl_word_t pack_ctrl_rsp_err(uint8_t err_u8) {
		ctrl_word_t w = 0;
		w.set_slc(0, (ac_int<4, false>)RSP_ERR);
		w.set_slc(8, (ac_int<8, false>)err_u8);
		return w;
	}

	static inline uint8_t unpack_ctrl_rsp_kind(const ctrl_word_t& w) {
		ac_int<4, false> k = w.template slc<4>(0);
		return (uint8_t)k.to_uint();
	}

	static inline uint8_t unpack_ctrl_rsp_payload(const ctrl_word_t& w) {
		ac_int<8, false> p = w.template slc<8>(8);
		return (uint8_t)p.to_uint();
	}

} // namespace aecct
