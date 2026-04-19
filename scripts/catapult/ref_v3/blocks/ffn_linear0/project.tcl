set refv3_block_project_script_dir [file dirname [file normalize [info script]]]
set refv3_block_name "RefV3FfnLinear0ReluBlock"
set refv3_block_project_name "Catapult_refv3_ffn_linear0"
set refv3_block_top "aecct_ref::ref_v3::RefV3FfnLinear0ReluBlockTop"
set refv3_block_filelist [file normalize [file join $refv3_block_project_script_dir "filelist.f"]]
set refv3_block_smoke_tb_rel "AECCT_ac_ref/tb_catapult/ref_v3/blocks/tb_ref_v3_ffn_linear0_relu_block_smoke.cpp"
source [file normalize [file join $refv3_block_project_script_dir ".." "common_ref_v3_block_project.tcl"]]

