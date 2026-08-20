#!/usr/bin/env bash
# mode.sh — one canonical mode predicate for shell-side evolution consumers.
#
# KG mode is selected only when knowledge_graph.json exists.  An ideas/
# directory by itself is not sufficient; this mirrors kg_ops.detect_mode().

if [[ -n "${LOOP_HOOK_MODE_SOURCED:-}" ]]; then
    return 0
fi
LOOP_HOOK_MODE_SOURCED=1

detect_evo_mode() {
    local evo_dir="$1"
    if [[ -f "${evo_dir}/knowledge_graph.json" ]]; then
        echo "kg"
    else
        echo "mcts"
    fi
}

is_kg_mode() {
    [[ "$(detect_evo_mode "$1")" == "kg" ]]
}

export -f detect_evo_mode is_kg_mode
