#!/usr/bin/env bash
# mode.sh — one canonical mode predicate for shell-side evolution consumers.
#
# New directories default to KG. Only a world_model.json-only directory is a
# legacy MCTS session. This mirrors kg_ops.detect_mode().

if [[ -n "${LOOP_HOOK_MODE_SOURCED:-}" ]]; then
    return 0
fi
LOOP_HOOK_MODE_SOURCED=1

detect_evo_mode() {
    local evo_dir="$1"
    if [[ -f "${evo_dir}/knowledge_graph.json" && -f "${evo_dir}/world_model.json" ]]; then
        echo "conflict"
        return 1
    elif [[ -f "${evo_dir}/world_model.json" ]]; then
        echo "mcts"
    else
        echo "kg"
    fi
}

is_kg_mode() {
    [[ "$(detect_evo_mode "$1")" == "kg" ]]
}

export -f detect_evo_mode is_kg_mode
