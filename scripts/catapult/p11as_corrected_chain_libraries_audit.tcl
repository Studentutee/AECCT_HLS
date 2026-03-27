set wrapper_sfd [file dirname [file normalize [info script]]]
set canonical_tcl [file normalize [file join $wrapper_sfd "p11as_corrected_chain_project.tcl"]]

proc p11as_audit_split_env_paths {raw_text} {
    set out {}
    foreach token [split $raw_text ";\n"] {
        set p [string trim $token]
        if {$p eq ""} { continue }
        lappend out $p
    }
    return $out
}

proc p11as_audit_set_option_path_list_optional {option_key paths} {
    set first 1
    foreach p $paths {
        if {$first} {
            if {[catch {options set $option_key $p} opt_err]} {
                puts "P11AS_AUDIT_OPTION_SKIP $option_key $opt_err"
                flush stdout
                return
            }
            set first 0
        } else {
            if {[catch {options set $option_key $p -append} opt_err]} {
                puts "P11AS_AUDIT_OPTION_SKIP $option_key $opt_err"
                flush stdout
                return
            }
        }
    }
}

if {![file exists $canonical_tcl]} {
    error "P11AS audit wrapper missing canonical Tcl: $canonical_tcl"
}

# Suppress canonical 'exit' so this audit-only wrapper can continue to go libraries.
rename exit __p11as_real_exit
proc exit args {
    puts "P11AS_AUDIT_SUPPRESS_EXIT"
    flush stdout
    return
}

source $canonical_tcl

rename exit {}
rename __p11as_real_exit exit

if {[info exists ::env(AECCT_P11AS_INPUT_LIBPATHS)]} {
    set audit_input_libpaths [p11as_audit_split_env_paths $::env(AECCT_P11AS_INPUT_LIBPATHS)]
    p11as_audit_set_option_path_list_optional "Input/LibPaths" $audit_input_libpaths
}
if {[info exists ::env(AECCT_P11AS_COMPONENTLIBS_SEARCHPATH)]} {
    set audit_component_search_paths [p11as_audit_split_env_paths $::env(AECCT_P11AS_COMPONENTLIBS_SEARCHPATH)]
    p11as_audit_set_option_path_list_optional "ComponentLibs/SearchPath" $audit_component_search_paths
}
if {[info exists ::env(AECCT_P11AS_TECHLIB_SEARCHPATHS)]} {
    set audit_techlib_paths [p11as_audit_split_env_paths $::env(AECCT_P11AS_TECHLIB_SEARCHPATHS)]
    p11as_audit_set_option_path_list_optional "ComponentLibs/TechLibSearchPath" $audit_techlib_paths
} elseif {[info exists ::env(AECCT_P11AS_COMPONENTLIBS_TECHLIBSEARCHPATH)]} {
    # Backward-compatible fallback for existing env naming in older wrappers.
    set audit_techlib_paths [p11as_audit_split_env_paths $::env(AECCT_P11AS_COMPONENTLIBS_TECHLIBSEARCHPATH)]
    p11as_audit_set_option_path_list_optional "ComponentLibs/TechLibSearchPath" $audit_techlib_paths
}

puts "P11AS_STAGE libraries START"
flush stdout
go libraries
puts "P11AS_STAGE libraries DONE"
flush stdout

exit
