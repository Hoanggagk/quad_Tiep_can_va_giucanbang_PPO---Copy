# Project instructions

## Language

- Communicate with the user in Vietnamese.
- Keep code identifiers and technical terminology in English.

## Project architecture

- `drone_control/algorithms/ppo.py`: shared PPO and Actor-Critic implementation.
- `drone_control/action_spaces/velocity_ned/`: 3D NED velocity method.
- `drone_control/action_spaces/motor_residual/`: 4-rotor residual motor method.
- `tools/`: visualization and simulator diagnostics.
- `analysis/legacy/`: legacy result-analysis scripts.
- `tests/`: test tree mirroring the source tree.

## Safety

- Do not start training or connect to the simulator unless explicitly requested.
- Do not overwrite checkpoints or experiment results.
- Preserve unrelated user changes.
- Before changing shared PPO or environment code, inspect its callers and impact.

## graphify

This project uses Graphify for codebase knowledge graphs. Generated graphs live in `graphify-out/`.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. When the answer is under a Markdown heading, `graphify section "<heading>"` prints that section's text, so the .md need not be opened. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
