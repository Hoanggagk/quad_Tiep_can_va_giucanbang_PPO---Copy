# Hướng dẫn sử dụng và chạy chương trình

Tài liệu này dùng cho cấu trúc mới của dự án PPO điều khiển
quadrotor. Tất cả lệnh được chạy bằng PowerShell từ **thư mục gốc
của repository**.

## 1. Cấu trúc chính

```text
drone_control/
├── algorithms/ppo.py                  # PPO/Actor-Critic dùng chung
└── action_spaces/
    ├── velocity_ned/                 # Action vận tốc NED, 3 chiều
    │   ├── env.py
    │   ├── train.py
    │   ├── control_projectairsim.py
    │   └── evaluate_balance.py
    └── motor_residual/               # Action residual RPM, 4 rotor
        ├── env.py
        ├── curriculum.py
        ├── calibrate.py
        ├── check_actuators.py
        └── train.py
tests/                               # Unit test
tools/visualization/                 # Vẽ log và kết quả
tools/diagnostics/                   # Chẩn đoán simulator/MAVSDK
config/                              # Calibration motor
sim_config/                          # Scene và robot ProjectAirSim
```

| Phương pháp | State | Action | Bounds |
|---|---:|---:|---|
| `velocity_ned` | 13 | 3 | `[-1.5, 1.5]^3` m/s |
| `motor_residual` | 17 | 4 | `[-1, 1]^4` |

Checkpoint của hai phương pháp không dùng lẫn nhau.

## 2. Chuẩn bị môi trường

```powershell
Set-Location "D:\thacsi\Control_drone\quad_Tiep_can_va_giucanbang_PPO - Copy"
$env:PYTHONUTF8 = "1"
python --version
python -c "import numpy, torch, matplotlib, pandas; print('Core dependencies: OK')"
python -c "import projectairsim; print('ProjectAirSim: OK')"
```

Chức năng MAVSDK cần package `mavsdk`. Một số diagnostic có thể cần
`opencv-python` hoặc `websocket-client`. ProjectAirSim và simulator phải được
cài theo môi trường của máy.

Luôn dùng `python -m ...` từ repository root; không chạy file con trực tiếp.

## 3. Kiểm tra tĩnh trước khi chạy simulator

```powershell
python -m unittest discover -v
python -m compileall -q drone_control tests tools analysis
```

Các lệnh trên không chạy training và không chủ động kết nối simulator.

## 4. NED velocity action space

Policy sinh action `[v_north, v_east, v_down]` theo m/s. Trong hệ NED,
`v_down < 0` nghĩa là bay lên.

### 4.1. Huấn luyện từ đầu

> Lệnh này kết nối ProjectAirSim và bắt đầu training.

```powershell
python -m drone_control.action_spaces.velocity_ned.train --help
python -m drone_control.action_spaces.velocity_ned.train
```

Output mặc định:

- Log CSV: `log_dir/PPO_log_hover_projectairsim.csv`.
- Checkpoint và learning curve: `log_dir/hover_projectairsim/`.

Nhánh velocity dùng tên output cố định. Hãy sao lưu artifact cũ trước
khi training lại nếu cần giữ kết quả.

### 4.2. Tiếp tục hoặc kiểm tra checkpoint

```powershell
python -m drone_control.action_spaces.velocity_ned.train `
  --resume "Ketquahuanluyen\3_4_2026\model\ep2006_step100000.pth"

python -m drone_control.action_spaces.velocity_ned.train `
  --test "Ketquahuanluyen\3_4_2026\model\ep2006_step100000.pth" `
  --num-test-episodes 5
```

Checkpoint phải có `state_dim=13` và `action_dim=3`.

### 4.3. Biến thể episode ngắn

Biến thể này dùng cùng action space, chỉ thay đổi horizon:

```powershell
python -m drone_control.action_spaces.velocity_ned.train_short_episode --help
python -m drone_control.action_spaces.velocity_ned.train_short_episode
```

### 4.4. Chạy policy trong ProjectAirSim

```powershell
python -m drone_control.action_spaces.velocity_ned.control_projectairsim `
  --model "Ketquahuanluyen\11_6_2026\ppo_hover_final.pth" `
  --episodes 5 `
  --save-dir "control_results\velocity_ned"
```

Dùng `--no-plot` nếu chỉ muốn lưu CSV.

### 4.5. Workflow có disturbance

```powershell
python -m drone_control.action_spaces.velocity_ned.control_disturbance `
  --model "Ketquahuanluyen\11_6_2026\ppo_hover_final.pth" `
  --episodes 5 `
  --save-dir "control_results\velocity_ned_disturbance"
```

Workflow này có thể phụ thuộc websocket/cơ chế disturbance của scene.

### 4.6. Đánh giá phục hồi cân bằng

```powershell
python -m drone_control.action_spaces.velocity_ned.evaluate_balance `
  --model "Ketquahuanluyen\3_4_2026\model\ep2006_step100000.pth" `
  --trials 3 `
  --save-dir "balance_eval_results"
```

Dùng `--no-plot` để bỏ qua bước vẽ. Output gồm CSV tổng hợp, trajectory
từng trial và recovery curve.

### 4.7. Điều khiển qua MAVSDK

Module hiện dùng hằng `CHECKPOINT_PATH` trong
`drone_control/action_spaces/velocity_ned/control_mavsdk.py`. Kiểm tra đường dẫn
trước khi chạy:

```powershell
python -m drone_control.action_spaces.velocity_ned.control_mavsdk
```

Lệnh có thể arm và điều khiển drone/PX4. Chỉ chạy trong môi trường
mô phỏng hoặc khu vực thử nghiệm đã kiểm soát.

## 5. Residual motor action space

Policy sinh bốn action theo thứ tự `[FL, FR, RL, RR]`. Environment ánh xạ
`[-1, 1]^4` thành residual RPM quanh operating point trong
`config/motor_control.json`.

### 5.1. Kiểm tra actuator

Mặc định lệnh sau là dry-run, không kết nối simulator và không gửi motor command:

```powershell
python -m drone_control.action_spaces.motor_residual.check_actuators
```

Smoke test thật, chỉ chạy sau khi đã xác nhận scene và actuator mapping:

```powershell
python -m drone_control.action_spaces.motor_residual.check_actuators `
  --execute `
  --signal 0.05 `
  --pulse-seconds 0.5 `
  --settle-seconds 0.25
```

`--signal` bị giới hạn tối đa `0.20`. Luôn bắt đầu từ giá trị thấp.

### 5.2. Hiệu chuẩn hover motor

```powershell
# Dry-run
python -m drone_control.action_spaces.motor_residual.calibrate

# Thực thi trong ProjectAirSim
python -m drone_control.action_spaces.motor_residual.calibrate `
  --execute `
  --output "config\motor_control.json"
```

Script từ chối ghi đè calibration đã tồn tại. Chỉ dùng `--overwrite` sau
khi đã sao lưu file cũ. Xem toàn bộ tham số bằng:

```powershell
python -m drone_control.action_spaces.motor_residual.calibrate --help
```

### 5.3. Huấn luyện motor-level PPO

> Lệnh này kết nối ProjectAirSim và bắt đầu training.

```powershell
python -m drone_control.action_spaces.motor_residual.train `
  --max-timesteps 500000 `
  --seed 42
```

Mỗi lần chạy tạo `run_id` duy nhất:

- Log: `log_dir/hover_motor/<run_id>/training.csv`.
- Checkpoint: `checkpoints_motor/<run_id>/`.
- Metadata: file `.pth.json` nằm cạnh mỗi checkpoint.

Workflow motor không ghi đè checkpoint đã tồn tại.

### 5.4. Resume motor-level training

```powershell
python -m drone_control.action_spaces.motor_residual.train `
  --resume "checkpoints_motor\<run_id>\episode_001000_step_000050000.pth" `
  --max-timesteps 100000 `
  --seed 42
```

Checkpoint bắt buộc có sidecar `.pth.json` hợp lệ, `state_dim=17`,
`action_dim=4` và đúng thứ tự actuator.

## 6. Vẽ và phân tích kết quả

```powershell
# Learning curve
python -m tools.visualization.visualize_results `
  --training-log "log_dir\PPO_log_hover_projectairsim.csv" `
  --smooth 20

# Trajectory
python -m tools.visualization.visualize_results `
  --trajectories "control_results\velocity_ned" `
  --goal 0 0 -2 `
  --goal-attitude 0 0 0

# So sánh hai nhóm
python -m tools.visualization.visualize_results `
  --compare "results\without_disturbance" "results\with_disturbance" `
  --compare-labels "Không nhiễu" "Có nhiễu" `
  --out-dir "results\comparison"

# Balance evaluation
python -m tools.visualization.visualize_results `
  --balance "balance_eval_results\balance_eval_summary.csv"
```

## 7. Thứ tự chạy khuyến nghị

### NED velocity

1. Chạy unit test.
2. Khởi động ProjectAirSim với scene NED velocity.
3. Chạy `velocity_ned.train` hoặc nạp checkpoint cũ.
4. Chạy `control_projectairsim` để kiểm tra policy.
5. Chạy `evaluate_balance` hoặc `control_disturbance`.
6. Dùng `tools.visualization` để tổng hợp kết quả.

### Residual motor

1. Chạy unit test.
2. Chạy `check_actuators` ở dry-run.
3. Smoke test với signal thấp trong scene motor.
4. Chạy `calibrate` ở dry-run, sau đó mới dùng `--execute`.
5. Xác minh `config/motor_control.json`.
6. Chạy `motor_residual.train`.
7. Giữ checkpoint cùng sidecar `.pth.json`.

## 8. Xử lý lỗi thường gặp

### `ModuleNotFoundError: drone_control`

Quay về repository root và dùng `python -m ...` thay vì chạy file con.

### `UnicodeEncodeError: cp1252`

```powershell
$env:PYTHONUTF8 = "1"
```

### Không nạp được checkpoint

- Bao đường dẫn trong dấu ngoặc kép.
- Không dùng checkpoint motor cho NED velocity hoặc ngược lại.
- Với motor resume, giữ file `.pth.json` cùng checkpoint.

### Simulator không phản hồi

- Kiểm tra ProjectAirSim đã chạy và scene trong `sim_config/` là chính xác.
- Kiểm tra tên drone `Drone1` và actuator ID của scene motor.
- Không chạy đồng thời nhiều workflow điều khiển cùng một drone.

## 9. Lưu ý an toàn

- Không chạy training, control, calibration hoặc smoke test khi chưa xác nhận
  simulator/thiết bị đích.
- Không thêm `--execute` nếu chỉ muốn kiểm tra cấu hình motor.
- Không dùng `--overwrite` nếu chưa sao lưu calibration cũ.
- Không ghi đè checkpoint hoặc kết quả thí nghiệm quan trọng.
- Khi dùng MAVSDK với drone thật, phải có emergency stop và người giám sát.
