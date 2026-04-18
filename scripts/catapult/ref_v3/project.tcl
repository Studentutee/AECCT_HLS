set sfd [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $sfd ".." ".." ".."]]
if {[info exists ::env(AECCT_REFV3_REPO_ROOT)] && $::env(AECCT_REFV3_REPO_ROOT) ne ""} {
    set repo_root [file normalize $::env(AECCT_REFV3_REPO_ROOT)]
}

set work_dir [file normalize [file join $repo_root "build" "ref_v3" "catapult_project"]]
if {[info exists ::env(AECCT_REFV3_CATAPULT_OUTDIR)] && $::env(AECCT_REFV3_CATAPULT_OUTDIR) ne ""} {
    set work_dir [file normalize $::env(AECCT_REFV3_CATAPULT_OUTDIR)]
}
file mkdir $work_dir
cd $work_dir

set refv3_top_entry "aecct_ref::ref_v3::RefV3CatapultTop"
set refv3_filelist [file normalize [file join $sfd "filelist.f"]]
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
set refv3_define_macros [list "REFV3_CATAPULT_MODE=1"]

proc refv3_set_option_path_list {option_key paths} {
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

proc refv3_resolve_repo_paths {repo_root rel_paths} {
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

if {![file exists $refv3_filelist]} {
    error "REFV3 filelist missing: $refv3_filelist"
}

puts "REFV3_CANONICAL_SYNTH_ENTRY $refv3_top_entry"
puts "REFV3_FILELIST $refv3_filelist"
flush stdout

options defaults
project new
options set Input/CppStandard c++20

set refv3_search_paths [refv3_resolve_repo_paths $repo_root $refv3_include_dirs]
refv3_set_option_path_list "Input/SearchPath" $refv3_search_paths

set refv3_compiler_flags ""
foreach d $refv3_define_macros {
    append refv3_compiler_flags " -D" $d
}
set refv3_compiler_flags [string trim $refv3_compiler_flags]
if {$refv3_compiler_flags ne ""} {
    options set Input/CompilerFlags $refv3_compiler_flags
}

set fd [open $refv3_filelist r]
while {[gets $fd line] >= 0} {
    set t [string trim $line]
    if {$t eq ""} { continue }
    if {[string index $t 0] eq "#"} { continue }

    set abs_path [file normalize [file join $repo_root $t]]
    if {![file exists $abs_path]} {
        close $fd
        error "REFV3 filelist entry missing: $abs_path"
    }
    solution file add $abs_path -type C++
}
close $fd

set refv3_e2e_tb [file normalize [file join $repo_root "AECCT_ac_ref" "tb_catapult" "ref_v3" "tb_ref_v3_catapult_e2e_4pattern.cpp"]]
if {[file exists $refv3_e2e_tb]} {
    solution file add $refv3_e2e_tb -type C++ -exclude true -description {C Testbench}
    puts "REFV3_EXCLUDED_TB $refv3_e2e_tb"
} else {
    puts "REFV3_EXCLUDED_TB_MISSING $refv3_e2e_tb"
}

solution design set $refv3_top_entry -top

puts "REFV3_STAGE compile START"
flush stdout
go compile
puts "REFV3_STAGE compile DONE"
flush stdout

exit
