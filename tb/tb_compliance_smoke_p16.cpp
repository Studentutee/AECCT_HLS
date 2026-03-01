#include <cstdint>
#include <cstdio>

#include "AecctProtocol.h"
#include "AecctTypes.h"
#include "design/AecctTop.h"

#if __has_include(<mc_scverify.h>)
#include <mc_scverify.h>
#endif

#ifndef CCS_MAIN
#define CCS_MAIN(...) int main(__VA_ARGS__)
#endif

#ifndef CCS_RETURN
#define CCS_RETURN(x) return (x)
#endif

class TbComplianceSmokeP16 {
public:
    int run_all() {
        aecct::AecctTop dut;
        aecct::ctrl_ch_t ctrl_cmd;
        aecct::ctrl_ch_t ctrl_rsp;
        aecct::data_ch_t data_in;
        aecct::data_ch_t data_out;

        ctrl_cmd.write(aecct::pack_ctrl_cmd((uint8_t)aecct::OP_SOFT_RESET));
        dut.run(ctrl_cmd, ctrl_rsp, data_in, data_out);

        aecct::u16_t rsp_word;
        if (!ctrl_rsp.nb_read(rsp_word)) {
            std::printf("ERROR: missing ctrl_rsp\n");
            return 1;
        }
        uint8_t kind = aecct::unpack_ctrl_rsp_kind(rsp_word);
        uint8_t payload = aecct::unpack_ctrl_rsp_payload(rsp_word);
        if (kind != (uint8_t)aecct::RSP_DONE || payload != (uint8_t)aecct::OP_SOFT_RESET) {
            std::printf("ERROR: unexpected ctrl_rsp kind=%u payload=%u\n", (unsigned)kind, (unsigned)payload);
            return 1;
        }

        std::printf("PASS: tb_compliance_smoke_p16\n");
        return 0;
    }
};

CCS_MAIN(int argc, char** argv) {
    (void)argc;
    (void)argv;
    TbComplianceSmokeP16 tb;
    int rc = tb.run_all();
    CCS_RETURN(rc);
}
