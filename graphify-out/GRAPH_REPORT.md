# Graph Report - quad_Tiep_can_va_giucanbang_PPO - Copy  (2026-11-17)

## Corpus Check
- 206 files · ~85,699 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 112 file(s) not represented in the graph (top: .csv 69, .pth 24, .urdf 6)
- WARNING: 112 of 318 files seen (35%) have no supported extractor and are missing from this graph entirely. It may look complete while representing only part of the codebase.

## Summary
- 1227 nodes · 2533 edges · 174 communities (33 shown, 141 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 183 edges (avg confidence: 0.89)
- Token cost: unavailable from the subagent runtime (stored counters: 0 input · 0 output)

## Community Hubs (Navigation)
- Math Module
- Actor Critic
- Calibrate Py
- Drone Model
- Learn Py
- Enums Py
- Motor Curriculum
- Matplotlib_Pyplot Module
- Drone Motor Hover Env
- Control_Mavsdk Py
- Base Aviary
- Argparse Module
- Numpy Module
- Steps Release Workflows
- Test Motor Reward
- Beta Aviary
- Base Control
- With_Disturbance Phantichketqua_Goc Py
- Drone Hovering Env
- Motor_Residual Env Py
- Step Env Motor_Residual
- Ctrl Aviary
- Phantichketquatest Py
- Ctbrcontrol Control
- Clock_Manual_Step Py
- Test Motor Action Mapping
- Os Module
- Steps Push Workflows
- Test_Env Py
- Step Env Velocity_Ned
- Ppo Algorithms
- Build_Evaluation_Manifest Evaluation_Protocol Drone_Control
- Legacy Phantichketqua_Goc Py
- Without_Disturbance Phantichketqua_Goc Py
- Success Tracker
- Project Agent Instructions
- Without_Disturbance Phantichketqua_Vitri Py
- Test Ppoaction Bounds Source Contract
- Ppo Actor Critic
- With_Disturbance Phantichketqua_Vitri Py
- Task Contract
- Compute_Gae Ppo Algorithms
- Residual Motor Action Space
- Ppo Quadcopter Hover Project
- Create_Evaluation_Manifest Py
- Two 64 Neuron Tanh Hidden
- Post Collision Balance Evaluation
- Env_Short_Episode Py
- Three Dimensional Ned Velocity Action
- Betaflight Sitl 4 5 Configuration
- Push East Recovery Curves
- Strong Push Recovery Curves
- Preserve Historical Experiment Artifacts For
- Late Altitude Divergence Under Disturbance
- Bounded Hover Tracking Under Disturbance
- Persistent Position Offsets
- Stable Altitude With Small Lateral
- Ppo Quadcopter Configuration Assessment
- Direct Four Actuator Reinforcement Learning
- Ppo Configuration Reference
- Build_Project Sh
- Clone_Bfs Sh
- Ffmpeg_Png2Mp4 Sh
- Multi Drone Helix Simulation Demo
- Py Bullet Grid Environment
- Likely Attitude Instability
- 4 R Disturbed Position Analysis
- 4 R Disturbed Flight Analysis
- Descent With Small Lateral And
- Descent With Bounded Lateral And
- Descent With Positive Yaw Drift
- Altitude Goal Tracking With Transient
- Altitude Goal Tracking With Repeated
- Altitude Goal Tracking With Multiple
- Repeatable Hover Response Across Episodes
- Undisturbed 4 R Attitude Analysis
- Undisturbed 4 R Episode 3
- Undisturbed 4 R Episode 6
- Undisturbed 4 R Episode 9
- Disturbed Episode Position Analysis
- Position And Attitude Response Metrics
- Actor Network Activations
- Actor Network Weight Distributions Across
- Actor Weight Heatmaps
- Diagonal Push Recovery Trials With
- Push Down Recovery Evaluation
- North Push Recovery Trials With
- Push South Recovery Evaluation
- Upward Push Recovery Trials With
- Push West Recovery Evaluation
- Ppo 3 R Disturbed Roll
- Noisy Ppo 3 R Trajectory
- Ppo 4 R Disturbed Roll
- Noisy Ppo 4 R Trajectory
- Ppo 3 R No Disturbance
- Noise Free Ppo 3 R
- Ppo 4 R No Disturbance
- Noise Free Ppo 4 R
- Drone Hovering Environment Explanation
- Project Usage Guide
- Motor Level Control Implementation Plan
- Ppo With Mavsdk Guide
- Multi Drone Helix Simulation State
- Multi Agent Drone Simulation Animation
- Gym Pybullet Drones Package Description
- Hover Training Reward Learning Curve
- Gym Pybullet Drones
- Ten Disturbed 4 R Episode
- Noisy 4 R Position Analysis
- Disturbed 4 R Episode 2
- Noisy 4 R Episode Overlay
- Disturbed 4 R Episode 1
- Noisy 4 R Trajectory Episode
- Disturbed 4 R Episode 4
- Trajectory_Ep005_Plot Noisy 4 R Trajectory
- Disturbed 4 R Episode 7
- Trajectory_Ep008_Plot Noisy 4 R Trajectory
- Disturbed 4 R Episode 10
- Noisy 4 R Raw Trajectory
- Disturbed 4 R Episode 3
- Trajectory_Ep004 Noisy 4 R Raw
- Disturbed 4 R Episode 6
- Trajectory_Ep007 Noisy 4 R Raw
- Disturbed 4 R Episode 9
- Trajectory_Ep010 Noisy 4 R Raw
- Ten No Disturbance 4 R
- Noise Free 4 R Position
- No Disturbance 4 R Episode
- Noise Free 4 R Trajectory
- Trajectory_Ep004 No Disturbance 4 R
- Trajectory_Ep005 Noise Free 4 R
- Trajectory_Ep007 No Disturbance 4 R
- Trajectory_Ep008 Noise Free 4 R
- Trajectory_Ep010 No Disturbance 4 R
- Noisy Position Analysis
- Disturbed Trajectory Attitude Rmse Peaks
- Noisy Attitude Analysis Episode 2

## God Nodes (most connected - your core abstractions)
1. `DroneMotorHoverEnv` - 54 edges
2. `BaseAviary` - 46 edges
3. `DroneModel` - 41 edges
4. `DroneHoveringEnv` - 34 edges
5. `Physics` - 33 edges
6. `PPO` - 31 edges
7. `MotorCurriculum` - 28 edges
8. `ActionType` - 25 edges
9. `ObservationType` - 25 edges
10. `Logger` - 22 edges

## Surprising Connections (you probably didn't know these)
- `TestMotorTrainingEntryPoint` --uses--> `MotorCurriculum`  [INFERRED]
  tests/action_spaces/motor_residual/test_train.py → drone_control/action_spaces/motor_residual/curriculum.py
- `TestMotorCurriculum` --uses--> `DroneMotorHoverEnv`  [INFERRED]
  tests/action_spaces/motor_residual/test_curriculum.py → drone_control/action_spaces/motor_residual/env.py
- `TestMotorActionMapping` --uses--> `DroneMotorHoverEnv`  [INFERRED]
  tests/action_spaces/motor_residual/test_env.py → drone_control/action_spaces/motor_residual/env.py
- `TestMotorObservation` --uses--> `DroneMotorHoverEnv`  [INFERRED]
  tests/action_spaces/motor_residual/test_env.py → drone_control/action_spaces/motor_residual/env.py
- `TestMotorRuntimeSafety` --uses--> `DroneMotorHoverEnv`  [INFERRED]
  tests/action_spaces/motor_residual/test_env.py → drone_control/action_spaces/motor_residual/env.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Shared PPO Across Action Spaces** — readme_ned_velocity_action_space, readme_residual_motor_action_space, readme_shared_ppo_implementation [EXTRACTED 1.00]
- **Residual Motor Training Pipeline** — drone_control_action_spaces_motor_residual_readme_hover_centered_mapping, drone_control_action_spaces_motor_residual_readme_motor_curriculum, drone_control_action_spaces_motor_residual_readme_checkpoint_metadata, drone_control_action_spaces_motor_residual_readme_training_monitor [EXTRACTED 1.00]

## Communities (174 total, 141 thin omitted)

### Community 0 - "Math Module"
Cohesion: 0.05
Nodes (31): _format_duration(), _mean(), _optional_float(), TrainingMonitor, build_checkpoint_metadata(), main(), make_run_id(), metadata_path_for() (+23 more)

### Community 1 - "Actor Critic"
Cohesion: 0.05
Nodes (10): ActorCritic, _normalize_action_bound(), RolloutBuffer, ScaleAction, TestActorCriticActionBounds, TestPPOConfigurations, TestScaleAction, main() (+2 more)

### Community 2 - "Calibrate Py"
Cohesion: 0.09
Nodes (30): binary_search_hover(), build_output(), CalibrationSettings, choose_best_trial(), equal_motor_command(), execute_calibration(), extract_z(), get_robot_metadata() (+22 more)

### Community 3 - "Drone Model"
Cohesion: 0.08
Nodes (4): DSLPIDControl, VelocityAviary, DroneModel, Physics

### Community 4 - "Learn Py"
Cohesion: 0.10
Nodes (12): run(), run(), run(), run(), run(), Logger, str2bool(), sync() (+4 more)

### Community 5 - "Enums Py"
Cohesion: 0.18
Nodes (12): BaseRLAviary, FlyThruGateAvitary, HoverAviary, MultiHoverAviary, ActionType, ImageType, ObservationType, PPO (+4 more)

### Community 6 - "Motor Curriculum"
Cohesion: 0.15
Nodes (6): CurriculumMetrics, CurriculumStage, CurriculumWindow, default_motor_curriculum(), MotorCurriculum, TestMotorCurriculum

### Community 7 - "Matplotlib_Pyplot Module"
Cohesion: 0.14
Nodes (13): _ensure_dir(), _load_traj_csv(), main(), parse_args(), plot_balance_summary(), plot_balance_trajectory(), plot_compare_folders(), _load_folder() (+5 more)

### Community 8 - "Drone Motor Hover Env"
Cohesion: 0.14
Nodes (5): EpisodeConditions, Box, DroneMotorHoverEnv, Env, load_motor_calibration()

### Community 9 - "Control_Mavsdk Py"
Cohesion: 0.11
Nodes (7): get_observation(), run(), save_trajectory(), get_observation(), monitor(), print_diagnostics(), run()

### Community 12 - "Argparse Module"
Cohesion: 0.12
Nodes (9): parse_args(), _plot_trajectory(), run_control(), _save_csv(), parse_args(), _plot_trajectory(), run_control(), _save_csv() (+1 more)

### Community 13 - "Numpy Module"
Cohesion: 0.17
Nodes (8): evaluate(), _plot_recovery_curves(), _plot_summary(), _print_table(), _run_trial(), _save_results_csv(), _save_trajectory_csv(), TrialResult

### Community 14 - "Steps Release Workflows"
Cohesion: 0.11
Nodes (21): needs, runs-on, steps, jobs, deploy, run_tests, name, on (+13 more)

### Community 18 - "With_Disturbance Phantichketqua_Goc Py"
Cohesion: 0.11
Nodes (7): compute_peak(), compute_recovery(), compute_time_in_band(), _mark_peak(), _shade_band(), _vline_collision(), _vline_settled()

### Community 19 - "Drone Hovering Env"
Cohesion: 0.16
Nodes (8): DroneHoveringEnv, main(), parse_args(), _plot_training_results(), _plot_trajectory(), _smooth(), test(), train()

### Community 24 - "Phantichketquatest Py"
Cohesion: 0.13
Nodes (4): _vline_pos(), _vline(), _vline_ang(), _vline_pos()

### Community 26 - "Clock_Manual_Step Py"
Cohesion: 0.20
Nodes (3): DroneWithDataSync, main(), nanos_to_sec()

### Community 28 - "Os Module"
Cohesion: 0.24
Nodes (7): main(), parse_args(), _plot_training_results(), _plot_trajectory_episode(), _smooth(), test(), train()

### Community 29 - "Steps Push Workflows"
Cohesion: 0.14
Nodes (13): jobs, run_tests, name, on, permissions, contents, runs-on, steps (+5 more)

### Community 34 - "Build_Evaluation_Manifest Evaluation_Protocol Drone_Control"
Cohesion: 0.29
Nodes (4): aggregate_metrics(), build_evaluation_manifest(), EpisodeMetrics, TestEvaluationProtocol

### Community 35 - "Legacy Phantichketqua_Goc Py"
Cohesion: 0.18
Nodes (4): compute_peak(), compute_time_in_band(), _mark_peak(), _shade_band()

### Community 36 - "Without_Disturbance Phantichketqua_Goc Py"
Cohesion: 0.18
Nodes (4): compute_peak(), compute_time_in_band(), _mark_peak(), _shade_band()

### Community 40 - "Project Agent Instructions"
Cohesion: 0.25
Nodes (7): Graphify-first Codebase Workflow, Action-space-oriented Project Architecture, Project Agent Instructions, 3D NED Velocity Action Space, Quadrotor Control with PPO, 4D Residual Motor Action Space, Shared PPO Implementation

### Community 41 - "Without_Disturbance Phantichketqua_Vitri Py"
Cohesion: 0.22
Nodes (3): compute_time_in_band(), _shade_band(), _vline_pos()

### Community 44 - "With_Disturbance Phantichketqua_Vitri Py"
Cohesion: 0.25
Nodes (3): _vline_collision(), _vline_pos(), _vline_settled()

### Community 48 - "Residual Motor Action Space"
Cohesion: 0.33
Nodes (5): Motor Checkpoint Metadata, Hover-centered Residual-to-motor Mapping, Motor Training Curriculum and Domain Randomization, Residual Motor Action Space, Motor Training Monitor Artifacts

### Community 54 - "Ppo Quadcopter Hover Project"
Cohesion: 0.33
Nodes (4): PPO Fly-through-gate Task, gym-pybullet-drones Environment, PPO Quadcopter Hover Project, PPO-PyTorch Algorithm

### Community 56 - "Two 64 Neuron Tanh Hidden"
Cohesion: 0.50
Nodes (4): Actor Network Architecture, Two 64-neuron Tanh Hidden Layers, 13-neuron Input State, 3-neuron Action Output

### Community 57 - "Post Collision Balance Evaluation"
Cohesion: 0.50
Nodes (4): Eight Disturbance Configurations, 0.3 m Position Error Threshold, Post-collision Balance Evaluation, Low Survival Across Disturbances

### Community 59 - "Three Dimensional Ned Velocity Action"
Cohesion: 0.50
Nodes (3): Three-dimensional NED velocity action space, Velocity policy training deployment and disturbance evaluation modules, NED velocity policy bounds [-1.5, 1.5]^3 m/s

### Community 60 - "Betaflight Sitl 4 5 Configuration"
Cohesion: 0.50
Nodes (4): Betaflight SITL 4.5 configuration, Betaflight pitch roll yaw PID profile, PWM motor protocol, Betaflight RC rate profile

### Community 64 - "Push East Recovery Curves"
Cohesion: 0.67
Nodes (3): Bounded Roll and Pitch Response, Large Post-impact Position Error, Push-east Recovery Curves

### Community 65 - "Strong Push Recovery Curves"
Cohesion: 0.67
Nodes (3): Strong Roll and Pitch Excursion, Severe Post-impact Position Error, Strong-push Recovery Curves

## Knowledge Gaps
- **125 isolated node(s):** `gym-pybullet-drones`, `Actor Network Activations`, `Actor Network Architecture`, `3-neuron Action Output`, `Actor network weight distributions across three layers` (+120 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 529 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **141 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseAviary` connect `Base Aviary` to `Drone Model`, `Enums Py`, `Beta Aviary Py`, `Reset Base Aviary Envs`, `_Get Drone State Vector`, `Beta Aviary`, `Step Base Aviary Envs`, `_Housekeeping Base Aviary Envs`, `_Export Image`, `Ctrl Aviary`, `_Preprocess Action`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `DroneMotorHoverEnv` (e.g. with `EpisodeConditions` and `MotorCurriculum`) actually correct?**
  _`DroneMotorHoverEnv` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `gym-pybullet-drones`, `Actor Network Activations`, `Actor Network Architecture` to the rest of the system?**
  _125 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Math Module` be split into smaller, more focused modules?**
  _Cohesion score 0.05143570536828964 - nodes in this community are weakly interconnected._
- **Why does `DroneMotorHoverEnv` connect `Drone Motor Hover Env` to `Math Module`, `Success Tracker`, `Motor Curriculum`, `Task Contract`, `Test Motor Reward`, `Close Env Motor_Residual`, `Motor_Residual Env Py`, `Step Env Motor_Residual`, `Test Motor Action Mapping`, `Test_Env Py`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `BaseAviary` (e.g. with `DroneModel` and `ImageType`) actually correct?**
  _`BaseAviary` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `Actor Critic` be split into smaller, more focused modules?**
  _Cohesion score 0.05191256830601093 - nodes in this community are weakly interconnected._
