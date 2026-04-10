#!/usr/bin/env sh

# --- Runner ---

# Build a prompt for the given skill, adapting for claude vs codex.
#   skill_prompt <skill-name> <claude-slash-command> [codex-description]
# Examples:
#   skill_prompt project-pipeline "/project-pipeline"       "pick and process the next Ready issue"
#   skill_prompt project-pipeline "/project-pipeline 97"    "process GitHub issue 97"
#   skill_prompt review-pipeline  "/review-pipeline"        "pick and process the next Review pool PR"
skill_prompt() {
    skill=$1
    slash_cmd=$2
    codex_desc=${3-}

    if [ "${RUNNER:-codex}" = "claude" ]; then
        echo "$slash_cmd"
    else
        echo "Use the repo-local skill at '.claude/skills/${skill}/SKILL.md'. Follow it to ${codex_desc}. Read the skill file directly instead of assuming Claude slash-command support."
    fi
}

# Build a prompt and optionally append structured context for Codex.
#   skill_prompt_with_context <skill> <slash-cmd> <codex-desc> <context-label> <context-json>
skill_prompt_with_context() {
    skill=$1
    slash_cmd=$2
    codex_desc=${3-}
    context_label=${4-}
    context_json=${5-}

    base_prompt=$(skill_prompt "$skill" "$slash_cmd" "$codex_desc")
    if [ "${RUNNER:-codex}" = "claude" ] || [ -z "$context_json" ]; then
        echo "$base_prompt"
    else
        printf '%s\n\n## %s\n%s\n' "$base_prompt" "$context_label" "$context_json"
    fi
}

# Run an agent with the configured runner (claude or codex).
#   run_agent <log-file> <prompt>
run_agent() {
    output_file=$1
    prompt=$2

    if [ "${RUNNER:-codex}" = "claude" ]; then
        claude --dangerously-skip-permissions \
            --model "${CLAUDE_MODEL:-opus}" \
            --verbose \
            --output-format text \
            --max-turns 500 \
            -p "$prompt" 2>&1 | tee "$output_file"
    else
        codex exec \
            --enable multi_agent \
            -m "${CODEX_MODEL:-gpt-5.4}" \
            -s danger-full-access \
            "$prompt" 2>&1 | tee "$output_file"
    fi
}
