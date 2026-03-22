set sfd [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $sfd ".." ".."]]
if {[info exists ::env(AECCT_P11AS_REPO_ROOT)] && $::env(AECCT_P11AS_REPO_ROOT) ne ""} {
    set repo_root [file normalize $::env(AECCT_P11AS_REPO_ROOT)]
}

set work_dir [file normalize [file join $repo_root "build" "p11as" "catapult_project"]]
if {[info exists ::env(AECCT_P11AS_CATAPULT_OUTDIR)] && $::env(AECCT_P11AS_CATAPULT_OUTDIR) ne ""} {
    set work_dir [file normalize $::env(AECCT_P11AS_CATAPULT_OUTDIR)]
}
file mkdir $work_dir
cd $work_dir

set p11as_top_entry "TopManagedAttentionChainCatapultTop::run"
set p11as_entry_tu [file normalize [file join $repo_root "src" "catapult" "p11as_top_managed_attention_chain_entry.cpp"]]

set p11as_include_dirs [list "." "include" "src" "gen/include" "third_party/ac_types" "data/weights"]
set p11as_define_macros [list "__SYNTHESIS__"]

proc p11as_split_env_paths {raw_text} {
    set out {}
    foreach token [split $raw_text ";\n"] {
        set p [string trim $token]
        if {$p eq ""} { continue }
        lappend out $p
    }
    return $out
}

proc p11as_resolve_repo_paths {repo_root rel_paths} {
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

proc p11as_set_option_path_list_required {option_key paths} {
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

proc p11as_set_option_path_list_optional {option_key paths} {
    set first 1
    foreach p $paths {
        if {$first} {
            if {[catch {options set $option_key $p} opt_err]} {
                puts "P11AS_OPTION_SKIP $option_key $opt_err"
                flush stdout
                return
            }
            set first 0
        } else {
            if {[catch {options set $option_key $p -append} opt_err]} {
                puts "P11AS_OPTION_SKIP $option_key $opt_err"
                flush stdout
                return
            }
        }
    }
}

proc p11as_set_single_option_if_env {option_key env_name} {
    if {![info exists ::env($env_name)]} { return }
    set v [string trim $::env($env_name)]
    if {$v eq ""} { return }
    if {[catch {options set $option_key $v} opt_err]} {
        puts "P11AS_OPTION_SKIP $option_key $opt_err"
        flush stdout
        return
    }
}

if {![file exists $p11as_entry_tu]} {
    error "P11AS entry TU missing: $p11as_entry_tu"
}

puts "P11AS_CANONICAL_SYNTH_ENTRY $p11as_top_entry"
puts "P11AS_ENTRY_TU $p11as_entry_tu"
flush stdout

options defaults
project new

options set Input/CppStandard c++20

set p11as_search_paths [p11as_resolve_repo_paths $repo_root $p11as_include_dirs]
p11as_set_option_path_list_required "Input/SearchPath" $p11as_search_paths

set p11as_compiler_flags ""
foreach d $p11as_define_macros {
    append p11as_compiler_flags " -D" $d
}
set p11as_compiler_flags [string trim $p11as_compiler_flags]
options set Input/CompilerFlags $p11as_compiler_flags

if {[info exists ::env(AECCT_P11AS_INPUT_LIBPATHS)]} {
    set p11as_input_libpaths [p11as_split_env_paths $::env(AECCT_P11AS_INPUT_LIBPATHS)]
    p11as_set_option_path_list_optional "Input/LibPaths" $p11as_input_libpaths
}
if {[info exists ::env(AECCT_P11AS_COMPONENTLIBS_SEARCHPATH)]} {
    set p11as_componentlib_search_paths [p11as_split_env_paths $::env(AECCT_P11AS_COMPONENTLIBS_SEARCHPATH)]
    p11as_set_option_path_list_optional "ComponentLibs/SearchPath" $p11as_componentlib_search_paths
}
if {[info exists ::env(AECCT_P11AS_COMPONENTLIBS_TECHLIBSEARCHPATH)]} {
    set p11as_componentlib_tech_paths [p11as_split_env_paths $::env(AECCT_P11AS_COMPONENTLIBS_TECHLIBSEARCHPATH)]
    p11as_set_option_path_list_optional "ComponentLibs/TechLibSearchPath" $p11as_componentlib_tech_paths
}

p11as_set_single_option_if_env "Flows/QuestaSIM/Path" "AECCT_P11AS_FLOWS_QUESTASIM_PATH"
p11as_set_single_option_if_env "Flows/DesignCompiler/Path" "AECCT_P11AS_FLOWS_DESIGNCOMPILER_PATH"
p11as_set_single_option_if_env "Flows/VSCode/INSTALL" "AECCT_P11AS_FLOWS_VSCODE_INSTALL"
p11as_set_single_option_if_env "Flows/VSCode/GDB_PATH" "AECCT_P11AS_FLOWS_VSCODE_GDB_PATH"

solution file add $p11as_entry_tu -type C++
solution design set $p11as_top_entry -top

puts "P11AS_STAGE compile START"
flush stdout
go compile
puts "P11AS_STAGE compile DONE"
flush stdout

exit
