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

# Kept for launch-pack checker continuity; intentionally not passed via -cflags in this minimal draft.
set p11as_include_dirs [list "." "include" "src" "gen/include" "third_party/ac_types" "data/weights"]
set p11as_define_macros [list "__SYNTHESIS__"]

if {![file exists $p11as_entry_tu]} {
    error "P11AS entry TU missing: $p11as_entry_tu"
}

puts "P11AS_CANONICAL_SYNTH_ENTRY $p11as_top_entry"
puts "P11AS_ENTRY_TU $p11as_entry_tu"
flush stdout

options defaults
project new
solution file add $p11as_entry_tu -type C++
solution design set $p11as_top_entry -top

puts "P11AS_STAGE compile START"
flush stdout
go compile
puts "P11AS_STAGE compile DONE"
flush stdout

exit
