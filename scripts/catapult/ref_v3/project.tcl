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

if {![file exists $refv3_filelist]} {
    error "REFV3 filelist missing: $refv3_filelist"
}

puts "REFV3_CANONICAL_SYNTH_ENTRY $refv3_top_entry"
puts "REFV3_FILELIST $refv3_filelist"
flush stdout

options defaults
project new
if {[catch {solution new ref_v3_compile} sol_new_err]} {
    puts "REFV3_INFO solution_new_fallback $sol_new_err"
    flush stdout
}

options set Input/CppStandard c++20

set refv3_search_paths [list \
    [file normalize $repo_root] \
    [file normalize [file join $repo_root "gen"]] \
    [file normalize [file join $repo_root "src"]] \
    [file normalize [file join $repo_root "AECCT_ac_ref"]] \
    [file normalize [file join $repo_root "AECCT_ac_ref" "include"]] \
    [file normalize [file join $repo_root "AECCT_ac_ref" "src"]] \
    [file normalize [file join $repo_root "AECCT_ac_ref" "catapult"]] \
    [file normalize [file join $repo_root "AECCT_ac_ref" "tb_catapult"]] \
    [file normalize [file join $repo_root "data" "weights"]] \
    [file normalize [file join $repo_root "third_party" "ac_types"]] \
]
refv3_set_option_path_list "Input/SearchPath" $refv3_search_paths
options set Input/CompilerFlags "-DREFV3_CATAPULT_MODE=1"

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

solution design set $refv3_top_entry -top

puts "REFV3_STAGE compile START"
flush stdout
go compile
puts "REFV3_STAGE compile DONE"
flush stdout

exit
