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

- `ppo.py`: PPO and Actor-Critic implementation.
- `envs.py`, `envs_100.py`: ProjectAirSim environments.
- `train_hover.py`, `train_hover_100.py`: training entry points.
- `ControlDrone*.py`: trained-policy control workflows.
- `evaluate_balance.py`: disturbance and recovery evaluation.

## Safety

- Do not start training or connect to the simulator unless explicitly requested.
- Do not overwrite checkpoints or experiment results.
- Preserve unrelated user changes.
- Before changing shared PPO or environment code, inspect its callers and impact.
