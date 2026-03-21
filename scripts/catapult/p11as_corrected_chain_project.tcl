set p11as_script_dir [file dirname [file normalize [info script]]]
set p11as_repo_root [file normalize [file join $p11as_script_dir ".." ".."]]
if {[info exists ::env(AECCT_P11AS_REPO_ROOT)] && $::env(AECCT_P11AS_REPO_ROOT) ne ""} {
    set p11as_repo_root [file normalize $::env(AECCT_P11AS_REPO_ROOT)]
}

set p11as_work_dir [file normalize [file join $p11as_repo_root "build" "p11as" "catapult_project"]]
if {[info exists ::env(AECCT_P11AS_CATAPULT_OUTDIR)] && $::env(AECCT_P11AS_CATAPULT_OUTDIR) ne ""} {
    set p11as_work_dir [file normalize $::env(AECCT_P11AS_CATAPULT_OUTDIR)]
}
file mkdir $p11as_work_dir

set p11as_project_name "p11as_corrected_chain"
set p11as_solution_name "solution1"
set p11as_top_entry "TopManagedAttentionChainCatapultTop::run"
set p11as_filelist [file normalize [file join $p11as_repo_root "scripts" "catapult" "p11as_corrected_chain_filelist.f"]]
set p11as_include_dirs [list "." "include" "src" "gen/include" "third_party/ac_types" "data/weights"]
set p11as_define_macros [list "__SYNTHESIS__"]

proc p11as_emit {msg} {
    puts $msg
    flush stdout
}

proc p11as_to_abs {repo rel} {
    if {[file pathtype $rel] eq "absolute"} {
        return [file normalize $rel]
    }
    return [file normalize [file join $repo $rel]]
}

proc p11as_build_cflags {repo include_dirs define_macros} {
    set flags "-std=c++14"
    foreach d $define_macros {
        append flags " -D" $d
    }
    foreach inc $include_dirs {
        append flags " -I\"" [p11as_to_abs $repo $inc] "\""
    }
    return [string trim $flags]
}

proc p11as_load_filelist {repo filelist_path} {
    if {![file exists $filelist_path]} {
        error "P11AS filelist missing: $filelist_path"
    }
    set fh [open $filelist_path r]
    set txt [read $fh]
    close $fh

    set out {}
    foreach raw [split $txt "\n"] {
        set line [string trim $raw]
        if {$line eq ""} { continue }
        if {[string first "#" $line] == 0} { continue }
        lappend out [p11as_to_abs $repo $line]
    }
    if {[llength $out] == 0} {
        error "P11AS filelist resolved to zero source files"
    }
    return $out
}

p11as_emit "P11AS_CANONICAL_SYNTH_ENTRY $p11as_top_entry"
p11as_emit "P11AS_PROJECT_DIR $p11as_work_dir"
p11as_emit "P11AS_FILELIST $p11as_filelist"

set p11as_cflags [p11as_build_cflags $p11as_repo_root $p11as_include_dirs $p11as_define_macros]
set p11as_sources [p11as_load_filelist $p11as_repo_root $p11as_filelist]

cd $p11as_work_dir
if {[catch {project new $p11as_project_name} p11as_project_err]} {
    error "P11AS project new failed: $p11as_project_err"
}
if {[catch {solution new $p11as_solution_name} p11as_solution_err]} {
    error "P11AS solution new failed: $p11as_solution_err"
}

foreach src $p11as_sources {
    if {![file exists $src]} {
        error "P11AS source missing: $src"
    }
    p11as_emit "P11AS_ADD_SOURCE $src"
    if {[catch {solution file add $src -type C++ -cflags $p11as_cflags} add_err]} {
        error "P11AS add source failed ($src): $add_err"
    }
}

if {[catch {solution design set $p11as_top_entry -top} settop_err]} {
    error "P11AS set top failed: $settop_err"
}

p11as_emit "P11AS_STAGE analyze START"
if {[catch {go analyze} analyze_err]} {
    error "P11AS analyze failed: $analyze_err"
}
p11as_emit "P11AS_STAGE analyze DONE"

p11as_emit "P11AS_STAGE compile START"
if {[catch {go compile} compile_err]} {
    error "P11AS compile failed: $compile_err"
}
p11as_emit "P11AS_STAGE compile DONE"

p11as_emit "P11AS_STAGE elaborate START"
if {[catch {go elaborate} elaborate_err]} {
    error "P11AS elaborate failed: $elaborate_err"
}
p11as_emit "P11AS_STAGE elaborate DONE"

p11as_emit "P11AS_STAGE architecture START"
if {[catch {go architecture} arch_err]} {
    p11as_emit "P11AS_STAGE architecture SKIP $arch_err"
} else {
    p11as_emit "P11AS_STAGE architecture DONE"
}

project save
p11as_emit "P11AS_PROJECT_SAVE DONE"
exit
