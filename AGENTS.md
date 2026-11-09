# Project instructions

## Language

- Communicate with the user in Vietnamese.
- Keep code identifiers and technical terminology in English.

## CodeGraph

A CodeGraph index is available in `.codegraph/`.

When locating or understanding code:

1. Use CodeGraph before grep, rg, find, or manually reading many files.
2. For architecture and behavioral questions, run:
   `codegraph explore "<question>"`
3. For a specific symbol or file, run:
   `codegraph node "<symbol-or-file>"`
4. For dependency analysis, use:
   `codegraph callers`, `codegraph callees`, and `codegraph impact`.
5. Fall back to direct file reading when CodeGraph output is insufficient.
6. After source-code changes, run:
   `codegraph sync .`

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
