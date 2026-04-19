if {![info exists refv3_block_project_script_dir]} {
    error "refv3_block_project_script_dir is required"
}
if {![info exists refv3_block_name]} {
    error "refv3_block_name is required"
}
if {![info exists refv3_block_project_name]} {
    error "refv3_block_project_name is required"
}
if {![info exists refv3_block_top]} {
    error "refv3_block_top is required"
}
if {![info exists refv3_block_filelist]} {
    error "refv3_block_filelist is required"
}
if {![info exists refv3_block_smoke_tb_rel]} {
    set refv3_block_smoke_tb_rel ""
}

proc refv3_block_set_option_path_list {option_key paths} {
    set first 1
    foreach p $paths {
        if {$first} {
            options set $option_key $p
            set first 0
        } else {
            options set $option_key $p -append
        }
    }
}

proc refv3_block_resolve_repo_paths {repo_root rel_paths} {
    set out {}
    foreach rel $rel_paths {
        if {$rel eq "."} {
            lappend out [file normalize $repo_root]
        } else {
            lappend out [file normalize [file join $repo_root $rel]]
        }
    }
    return $out
}

proc refv3_block_run_stage {stage_name stage_cmd} {
    puts "REFV3_BLOCK_STAGE $stage_name START"
    flush stdout
    if {[catch {uplevel #0 $stage_cmd} stage_err]} {
        puts "REFV3_BLOCK_STAGE $stage_name FAIL $stage_err"
        flush stdout
        return 0
    }
    puts "REFV3_BLOCK_STAGE $stage_name DONE"
    flush stdout
    return 1
}

proc refv3_block_status_from_bool {ok} {
    if {$ok} {
        return "OK"
    }
    return "FAIL"
}

set repo_root [file normalize [file join $refv3_block_project_script_dir ".." ".." ".." ".." ".."]]
if {[info exists ::env(AECCT_REFV3_REPO_ROOT)] && $::env(AECCT_REFV3_REPO_ROOT) ne ""} {
    set repo_root [file normalize $::env(AECCT_REFV3_REPO_ROOT)]
}

set work_dir [file normalize [file join $repo_root "build" "ref_v3" "blocks" $refv3_block_project_name "project"]]
if {[info exists ::env(AECCT_REFV3_BLOCKS_OUTROOT)] && $::env(AECCT_REFV3_BLOCKS_OUTROOT) ne ""} {
    set work_dir [file normalize [file join $::env(AECCT_REFV3_BLOCKS_OUTROOT) $refv3_block_project_name "project"]]
}
if {[info exists ::env(AECCT_REFV3_BLOCK_CATAPULT_OUTDIR)] && $::env(AECCT_REFV3_BLOCK_CATAPULT_OUTDIR) ne ""} {
    set work_dir [file normalize $::env(AECCT_REFV3_BLOCK_CATAPULT_OUTDIR)]
}

file mkdir $work_dir
cd $work_dir

set refv3_include_dirs [list \
    "." \
    "gen" \
    "src" \
    "AECCT_ac_ref" \
    "AECCT_ac_ref/include" \
    "AECCT_ac_ref/src" \
    "AECCT_ac_ref/catapult" \
    "AECCT_ac_ref/tb_catapult" \
    "data/weights" \
    "third_party/ac_types" \
]
set refv3_define_macros [list \
    "REFV3_CATAPULT_MODE=1" \
    "AECCT_REFV3_CATAPULT_COMPILE_STUB=1" \
]

set refv3_block_filelist_abs [file normalize $refv3_block_filelist]
if {![file exists $refv3_block_filelist_abs]} {
    error "REFV3 block filelist missing: $refv3_block_filelist_abs"
}

set refv3_block_smoke_tb_abs ""
if {$refv3_block_smoke_tb_rel ne ""} {
    set refv3_block_smoke_tb_abs [file normalize [file join $repo_root $refv3_block_smoke_tb_rel]]
}

puts "REFV3_BLOCK_NAME $refv3_block_name"
puts "REFV3_BLOCK_PROJECT_NAME $refv3_block_project_name"
puts "REFV3_BLOCK_TOP $refv3_block_top"
puts "REFV3_BLOCK_FILELIST $refv3_block_filelist_abs"
puts "REFV3_BLOCK_WORK_DIR $work_dir"
flush stdout

options defaults
project new
options set Input/CppStandard c++20

set refv3_search_paths [refv3_block_resolve_repo_paths $repo_root $refv3_include_dirs]
refv3_block_set_option_path_list "Input/SearchPath" $refv3_search_paths

set refv3_compiler_flags ""
foreach d $refv3_define_macros {
    append refv3_compiler_flags " -D" $d
}
set refv3_compiler_flags [string trim $refv3_compiler_flags]
if {$refv3_compiler_flags ne ""} {
    options set Input/CompilerFlags $refv3_compiler_flags
}

set fd [open $refv3_block_filelist_abs r]
while {[gets $fd line] >= 0} {
    set t [string trim $line]
    if {$t eq ""} { continue }
    if {[string index $t 0] eq "#"} { continue }

    set abs_path [file normalize [file join $repo_root $t]]
    if {![file exists $abs_path]} {
        close $fd
        error "REFV3 block filelist entry missing: $abs_path"
    }
    solution file add $abs_path -type C++
}
close $fd

if {$refv3_block_smoke_tb_abs ne ""} {
    if {[file exists $refv3_block_smoke_tb_abs]} {
        solution file add $refv3_block_smoke_tb_abs -type C++ -exclude true -description {C Testbench}
        puts "REFV3_BLOCK_EXCLUDED_TB $refv3_block_smoke_tb_abs"
    } else {
        puts "REFV3_BLOCK_EXCLUDED_TB_MISSING $refv3_block_smoke_tb_abs"
    }
}
flush stdout

solution design set $refv3_block_top -top

set stage_compile_ok [refv3_block_run_stage "compile" "go compile"]
set stage_compile_status [refv3_block_status_from_bool $stage_compile_ok]

set stage_libraries_status "SKIP"
set stage_libraries_ok 0
if {$stage_compile_ok} {
    set stage_libraries_ok [refv3_block_run_stage "libraries" "go libraries"]
    set stage_libraries_status [refv3_block_status_from_bool $stage_libraries_ok]
}

set stage_assembly_status "SKIP"
set stage_assembly_ok 0
if {$stage_libraries_ok} {
    set stage_assembly_ok [refv3_block_run_stage "assembly" "go assembly"]
    set stage_assembly_status [refv3_block_status_from_bool $stage_assembly_ok]
}

set stage_architect_status "SKIP"
set stage_architect_ok 0
if {$stage_assembly_ok} {
    set stage_architect_ok [refv3_block_run_stage "architect" "go architect"]
    set stage_architect_status [refv3_block_status_from_bool $stage_architect_ok]
}

set stage_allocate_status "SKIP"
set stage_allocate_ok 0
if {$stage_architect_ok} {
    set stage_allocate_ok [refv3_block_run_stage "allocate" "go allocate"]
    set stage_allocate_status [refv3_block_status_from_bool $stage_allocate_ok]
}

set stage_extract_status "SKIP"
set stage_extract_ok 0
if {$stage_allocate_ok} {
    set stage_extract_ok [refv3_block_run_stage "extract" "go extract"]
    set stage_extract_status [refv3_block_status_from_bool $stage_extract_ok]
}

set stage_project_save_ok [refv3_block_run_stage "project_save" "project save"]
set stage_project_save_status [refv3_block_status_from_bool $stage_project_save_ok]

set ccs_paths [glob -nocomplain -types f [file join $work_dir "*.ccs"]]
set solution_dirs [glob -nocomplain -types d [file join $work_dir "solution*"]]
set directives_path [file join $work_dir "directives.tcl"]
set messages_path [file join $work_dir "messages.txt"]
set primary_solution_dir ""
if {[llength $solution_dirs] > 0} {
    set primary_solution_dir [lindex $solution_dirs 0]
}

set report_hint_area ""
set report_hint_power ""
set report_hint_perf ""
if {$primary_solution_dir ne ""} {
    set report_hint_area [file join $primary_solution_dir "syn" "report"]
    set report_hint_power [file join $primary_solution_dir "power" "report"]
    set report_hint_perf [file join $primary_solution_dir "rtl" "report"]
}

set manifest_path [file join $work_dir "refv3_block_manifest.txt"]
set mfd [open $manifest_path w]
puts $mfd "block_name=$refv3_block_name"
puts $mfd "project_name=$refv3_block_project_name"
puts $mfd "top_name=$refv3_block_top"
puts $mfd "repo_root=$repo_root"
puts $mfd "work_dir=$work_dir"
puts $mfd "filelist=$refv3_block_filelist_abs"
puts $mfd "smoke_tb=$refv3_block_smoke_tb_abs"
puts $mfd "stage_compile=$stage_compile_status"
puts $mfd "stage_libraries=$stage_libraries_status"
puts $mfd "stage_assembly=$stage_assembly_status"
puts $mfd "stage_architect=$stage_architect_status"
puts $mfd "stage_allocate=$stage_allocate_status"
puts $mfd "stage_extract=$stage_extract_status"
puts $mfd "stage_project_save=$stage_project_save_status"
puts $mfd "ccs_paths=[join $ccs_paths {;}]"
puts $mfd "solution_dirs=[join $solution_dirs {;}]"
puts $mfd "directives_path=$directives_path"
puts $mfd "messages_path=$messages_path"
puts $mfd "report_hint_area=$report_hint_area"
puts $mfd "report_hint_power=$report_hint_power"
puts $mfd "report_hint_perf=$report_hint_perf"
puts $mfd "governance_posture=not Catapult closure; not SCVerify closure"
close $mfd

puts "REFV3_BLOCK_MANIFEST $manifest_path"
puts "REFV3_BLOCK_PROJECT_SAVE_PATH $work_dir"
puts "REFV3_BLOCK_REPORT_HINT_AREA $report_hint_area"
puts "REFV3_BLOCK_REPORT_HINT_POWER $report_hint_power"
puts "REFV3_BLOCK_REPORT_HINT_PERF $report_hint_perf"
flush stdout

if {!$stage_project_save_ok} {
    exit 3
}
if {!$stage_compile_ok} {
    exit 2
}

exit 0
