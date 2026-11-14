# Bảng Tham Số Cấu Hình Thuật Toán PPO

> Các mục 1–9 bên dưới mô tả cấu hình `velocity_ned` với 3 hành động vận tốc.
> Cấu hình PPO điều khiển trực tiếp 4 động cơ (`motor_residual`) được trình bày
> riêng tại [Mục 10](#10-cấu-hình-ppo-cho-không-gian-hành-động-4-động-cơ).

## 1. Tham Số Không Gian (State & Action Space)

| Tham Số        | Ký Hiệu | Giá Trị     | Ý Nghĩa                                                                                                             |
| -------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| `STATE_DIM`    | $N_s$   | 13          | Chiều không gian trạng thái: [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal] |
| `ACTION_DIM`   | $N_a$   | 3           | Chiều không gian hành động: [v_north, v_east, v_down]                                                               |
| `Action Range` | $A$     | [-1.5, 1.5] | c động                                                                                           |

---

## 2. Tham Số Độ Lệch Chuẩn Hành Động (Action Standard Deviation)

| Tham Số                 | Ký Hiệu        | Giá Trị     | Ý Nghĩa                                                             |
| ----------------------- | -------------- | ----------- | ------------------------------------------------------------------- |
| `ACTION_STD_INIT`       | $\sigma_0$     | 0.5         | Độ lệch chuẩn ban đầu (~33% của range ±1.5)                         |
| `MIN_ACTION_STD`        | $\sigma_{min}$ | 0.01        | Độ lệch chuẩn tối thiểu (~1.7% của range, ≈±0.15 m/s khi khai thác) |
| `ACTION_STD_DECAY_RATE` | $\beta$        | 0.01        | Tốc độ giảm std mỗi lần giảm                                        |
| `ACTION_STD_DECAY_FREQ` | $F_\sigma$     | 50 episodes | Tần suất giảm action_std (cứ 50 episode)                            |

**Giải thích**: Độ lệch chuẩn cao ban đầu giúp khám phá (exploration), sau đó giảm dần để khai thác (exploitation).

---

## 3. Tham Số Vòng Lặp Huấn Luyện (Training Loop)

| Tham Số                  | Ký Hiệu   | Giá Trị         | Ý Nghĩa                                              |
| ------------------------ | --------- | --------------- | ---------------------------------------------------- |
| `MAX_TRAINING_TIMESTEPS` | $T_{max}$ | 5×10⁵ (500,000) | Tổng số timestep tối đa                              |
| `STEPS_PER_EPISODE`      | $S_e$     | 50              | Số bước tối đa mỗi episode (50 × 0.1s = 5s thực)     |
| `UPDATE_TIMESTEP`        | $T_{up}$  | 2,000           | Cập nhật PPO sau mỗi N timestep                      |
| `K_EPOCHS`               | $K$       | 30              | Số epoch per update (lặp trên dữ liệu trong 1 batch) |

**Giải thích**:

- Mỗi episode chịu tối đa 50 bước
- Sau mỗi 2,000 timestep (≈40 episodes), cập nhật network PPO
- Mỗi lần update, lặp 30 lần trên dữ liệu batch

---

## 4. Tham Số Thuật Toán PPO (PPO Algorithm)

| Tham Số     | Ký Hiệu    | Giá Trị         | Ý Nghĩa                                                |
| ----------- | ---------- | --------------- | ------------------------------------------------------ |
| `EPS_CLIP`  | $\epsilon$ | 0.2             | Clip hệ số: π_new / π_old ∈ [1-ε, 1+ε] = [0.8, 1.2]    |
| `GAMMA`     | $\gamma$   | 0.99            | Hệ số chiết khấu (discount factor) - giá trị tương lai |
| `LR_ACTOR`  | $\alpha_a$ | 3×10⁻⁴ (0.0003) | Learning rate cho Actor network                        |
| `LR_CRITIC` | $\alpha_c$ | 1×10⁻³ (0.001)  | Learning rate cho Critic network                       |

**Giải thích**:

- `EPS_CLIP = 0.2`: Hạn chế mức độ thay đổi policy trong mỗi update (tránh thay đổi quá tột)
- `GAMMA = 0.99`: Cân bằng giữa reward hiện tại và tương lai
- `LR_CRITIC > LR_ACTOR`: Critic học nhanh hơn để ước lượng value chính xác

---

## 5. Tham Số Kiểm Soát Log (Logging & Checkpointing)

| Tham Số           | Ký Hiệu | Giá Trị               | Ý Nghĩa                             |
| ----------------- | ------- | --------------------- | ----------------------------------- |
| `PRINT_FREQ`      | $F_p$   | 1,000 timesteps       | In log console mỗi N timestep       |
| `LOG_FREQ`        | $F_l$   | 250 timesteps         | Ghi CSV log mỗi N timestep          |
| `SAVE_MODEL_FREQ` | $F_s$   | 50,000 timesteps      | Lưu checkpoint mỗi N timestep       |
| `LOG_DIR`         | —       | "log_dir/"            | Thư mục lưu trữ logs và checkpoints |
| `RUN_NAME`        | —       | "hover_projectairsim" | Tên run (tạo subfolder)             |

---

## 6. Tham Số Môi trường (Environment)

| Tham Số                 | Ký Hiệu      | Giá Trị     | Ý Nghĩa                                   |
| ----------------------- | ------------ | ----------- | ----------------------------------------- |
| `dt`                    | $\Delta t$   | 0.1 seconds | Thời gian giữa các lần gửi lệnh hành động |
| `MAX_STEPS_PER_EPISODE` | $S_{max}$    | 50 bước     | = 50 × 0.1s = 5s thực                     |
| `Goal Pose`             | $\mathbf{g}$ | [0, 0, -2]  | Mục tiêu bay: NED (North, East, Down)     |

---

## 7. Tham Số Reward Function

| Tham Số             | Ký Hiệu    | Giá Trị | Ý Nghĩa                                                  |
| ------------------- | ---------- | ------- | -------------------------------------------------------- |
| `r_pos` weight      | $w_p$      | 1.0     | Thành phần vị trí (1.0 khi ở tâm, giảm đến 0 nếu quá xa) |
| `r_approach` weight | $w_a$      | 2.0     | Thưởng khi tiếp cận mục tiêu                             |
| `r_att` weight      | $w_{att}$  | -0.1    | Phạt nếu drone nghiêng quá nhiều                         |
| `survival bonus`    | $r_s$      | 0.01    | Thưởng sống sót mỗi bước                                 |
| `hover bonus`       | $r_h$      | 0.5     | Thưởng khi hover ở vị trí (<0.2m, <0.2m/s)               |
| `success bonus`     | $r_{succ}$ | 5.0     | Thưởng khi hover thành công >20 bước                     |
| `crash penalty`     | $r_c$      | -2.0    | Phạt khi va chạm hoặc quá xa                             |
| `ground penalty`    | $r_g$      | -2.0    | Phạt khi chạm mặt đất                                    |

---

## 8. Tóm Tắt Nhanh

### Network Architecture (ActorCritic)

```
Actor Network:
  Input (13D) → LayerNorm → 64 → Tanh → 64 → Tanh → 3 → Tanh → Scale(1.5)

Critic Network:
  Input (13D) → LayerNorm → 64 → Tanh → 64 → Tanh → 1 (Value)
```

### Timeline Huấn Luyện

- **500,000 timestep** tổng cộng
- ≈ 10,000 episodes (mỗi 50 timestep)
- ≈ 250 updates (mỗi 2,000 timestep)
- Mỗi update: 30 epochs trên ~2,000 timestep dữ liệu

---

## 9. Các Biến Thể Cấu Hình (Dành Tham Khảo)

| Tình Huống             | Thay Đổi                                             |
| ---------------------- | ---------------------------------------------------- |
| **Học nhanh hơn**      | Tăng `LR_ACTOR`, `LR_CRITIC`; giảm `K_EPOCHS`        |
| **Ổn định hơn**        | Giảm `LR_ACTOR`, `LR_CRITIC`; tăng `K_EPOCHS`        |
| **Khám phá nhiều hơn** | Tăng `ACTION_STD_INIT`; giảm `ACTION_STD_DECAY_RATE` |
| **Khai thác tốt hơn**  | Giảm `ACTION_STD_INIT`; tăng `ACTION_STD_DECAY_RATE` |
| **Tránh overfit**      | Giảm `K_EPOCHS`; tăng `EPS_CLIP`                     |

---

## 10. Cấu hình PPO cho không gian hành động 4 động cơ

### 10.1. Phạm vi và file cấu hình

Cấu hình này thuộc module `drone_control.action_spaces.motor_residual`. Policy
không phát lệnh vận tốc NED mà sinh bốn residual action độc lập quanh điểm làm
việc hover đã hiệu chuẩn.

| Thành phần | File nguồn |
|---|---|
| PPO và Actor-Critic dùng chung | `drone_control/algorithms/ppo.py` |
| Siêu tham số và vòng lặp huấn luyện | `drone_control/action_spaces/motor_residual/train.py` |
| State, action, reward và termination | `drone_control/action_spaces/motor_residual/env.py` |
| Curriculum và domain randomization | `drone_control/action_spaces/motor_residual/curriculum.py` |
| Hiệu chuẩn operating point | `config/motor_control.json` |

Không dùng checkpoint của `velocity_ned` cho cấu hình này vì kích thước state,
action và ý nghĩa đầu ra policy khác nhau.

### 10.2. Không gian trạng thái

`STATE_DIM = 17`. Thứ tự các phần tử phải được giữ nguyên khi huấn luyện,
resume hoặc triển khai policy:

| Chỉ số | Tên trong metadata | Đại lượng |
|---:|---|---|
| 0–2 | `x_m`, `y_m`, `z_m` | Vị trí NED, đơn vị m |
| 3–5 | `roll_rad`, `pitch_rad`, `yaw_rad` | Góc Euler, đơn vị rad |
| 6–8 | `vx_mps`, `vy_mps`, `vz_mps` | Vận tốc tuyến tính, đơn vị m/s |
| 9–11 | `roll_rate_radps`, `pitch_rate_radps`, `yaw_rate_radps` | Vận tốc góc, đơn vị rad/s |
| 12 | `distance_to_goal_m` | Khoảng cách Euclid tới mục tiêu, đơn vị m |
| 13–16 | `motor_fl_signal`, `motor_fr_signal`, `motor_rl_signal`, `motor_rr_signal` | Control signal hiện tại của bốn động cơ |

`LayerNorm(17)` được áp dụng trước Actor và Critic trong implementation PPO dùng
chung. Việc đổi thứ tự hoặc bỏ các motor signal sẽ làm checkpoint hiện tại
không còn tương thích.

### 10.3. Không gian hành động và ánh xạ xuống động cơ

`ACTION_DIM = 4`, theo đúng thứ tự:

```text
[FL, FR, RL, RR]
```

Tương ứng với các actuator:

```text
[Prop_FL_actuator, Prop_FR_actuator, Prop_RL_actuator, Prop_RR_actuator]
```

Mỗi action được giới hạn trong `[-1, 1]`. Với action chuẩn hóa `a_i`, environment
tính lệnh cho động cơ thứ `i` như sau:

```text
rpm_i = clip(hover_rpm + a_i * rpm_delta, min_rpm, max_rpm)
control_signal_i = clip((rpm_i / max_rpm)^2 + hover_signal_bias, 0, 1)
```

Trong đó `hover_rpm`, `max_rpm` và thứ tự actuator được đọc từ
`config/motor_control.json`; `rpm_delta` được chọn sao cho residual đối xứng
quanh `hover_rpm` mà không vượt giới hạn RPM. Vì vậy `a_i = 0` là operating
point hover, `a_i < 0` giảm RPM và `a_i > 0` tăng RPM. Cần hiệu chuẩn lại khi
đổi model động cơ, khối lượng, thrust model hoặc scene vật lý.

### 10.4. Siêu tham số PPO mặc định

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `ACTION_STD_INIT` | 0.35 | Độ lệch chuẩn khám phá ban đầu trên action chuẩn hóa |
| `ACTION_STD_DECAY_RATE` | 0.01 | Mức giảm mỗi lần decay |
| `ACTION_STD_DECAY_FREQ_EPISODES` | 50 episode | Chu kỳ giảm độ lệch chuẩn |
| `MIN_ACTION_STD` | 0.05 | Độ lệch chuẩn nhỏ nhất |
| `MAX_TRAINING_TIMESTEPS` | 500,000 | Tổng số bước huấn luyện mặc định |
| `STEPS_PER_EPISODE` | 500 | Horizon tối đa của một episode |
| `UPDATE_TIMESTEP` | 2,000 | Số transition giữa hai lần PPO update |
| `K_EPOCHS` | 30 | Số epoch tối ưu trên mỗi rollout batch |
| `EPS_CLIP` | 0.2 | Ngưỡng PPO clipped objective |
| `GAMMA` | 0.99 | Hệ số chiết khấu |
| `LR_ACTOR` | 3×10⁻⁴ | Learning rate của Actor |
| `LR_CRITIC` | 1×10⁻³ | Learning rate của Critic |
| `SAVE_MODEL_FREQ` | 50,000 timestep | Chu kỳ lưu checkpoint |

Với `CONTROL_FREQUENCY_HZ = 50`, ta có `dt = 0.02 s`; horizon 500 bước tương
đương tối đa 10 giây mô phỏng. Một PPO update nhận tối đa khoảng bốn episode
đủ horizon, nhưng số episode thực tế có thể lớn hơn do termination sớm.

### 10.5. Kiến trúc Actor-Critic

```text
State 17D → LayerNorm
  ├─ Actor : 17 → 64 → Tanh → 64 → Tanh → 4 → Tanh → Scale[-1, 1]
  └─ Critic: 17 → 64 → Tanh → 64 → Tanh → 1
```

Actor sử dụng Gaussian policy với phương sai chéo; action lấy mẫu được clamp về
`[-1, 1]^4` trước khi gửi sang environment.

### 10.6. Reward mặc định

Reward được scale theo `dt / reference_dt`, với `reference_dt = 0.02 s`, để giữ
ý nghĩa gần nhất quán khi thay đổi tần số điều khiển. Cấu hình mặc định gồm:

| Thành phần | Trọng số | Vai trò |
|---|---:|---|
| `position` | `position_weight = 1.0`, `position_scale = 0.5` | Thưởng theo `exp(-0.5 * position_error²)` |
| `velocity` | 0.10 | Phạt bình phương tốc độ tuyến tính |
| `attitude` | 0.50 | Phạt bình phương roll và pitch |
| `yaw` | 0.05 | Phạt bình phương sai lệch yaw về 0 |
| `angular_rate` | 0.05 | Phạt bình phương tốc độ góc |
| `motor_effort` | 0.02 | Phạt lệch motor signal khỏi operating point hover |
| `motor_smoothness` | 0.10 | Phạt thay đổi motor signal giữa hai bước |
| `saturation` | 0.20 | Phạt tỷ lệ động cơ gần biên `[0, 1]` |
| `alive_bonus` | 0.01 | Thưởng duy trì episode |
| `hover_bonus` | 0.50 | Thưởng khi đồng thời đạt điều kiện hover |
| `crash_penalty` | 10.0 | Phạt một lần khi trạng thái được đánh dấu crash |

Điều kiện nhận `hover_bonus` là position error `< 0.20 m`, speed `< 0.20 m/s`
và chuẩn attitude roll-pitch `< 10°`. Không nên thay một trọng số riêng lẻ mà
không theo dõi ít nhất reward từng thành phần, survival rate, position RMSE,
attitude RMSE và saturation fraction.

### 10.7. Termination và curriculum

Episode kết thúc sớm khi xảy ra một trong các điều kiện:

- collision;
- `|roll|` hoặc `|pitch|` vượt giới hạn attitude của environment;
- chạm gần mặt đất (`z > -0.2 m` trong hệ NED);
- khoảng cách tới mục tiêu lớn hơn `5 m`;
- chuẩn vận tốc lớn hơn `5 m/s`.

Curriculum mặc định tiến qua bảy stage:

```text
attitude_hover → random_attitude → random_position_velocity
→ altitude_change → takeoff → disturbance → domain_randomization
```

Stage chỉ được nâng khi đủ số episode đánh giá và đồng thời đạt các ngưỡng về
survival rate, position RMSE và attitude RMSE. Khi resume, stage hiện tại được
khôi phục từ metadata của checkpoint.

### 10.8. Logging, checkpoint và tương thích

Mỗi lần chạy tạo `run_id` riêng:

| Artifact | Vị trí |
|---|---|
| Log từng episode | `log_dir/hover_motor/<run_id>/training.csv` |
| Trạng thái monitor mới nhất | `log_dir/hover_motor/<run_id>/training_status.json` |
| Checkpoint | `checkpoints_motor/<run_id>/*.pth` |
| Metadata checkpoint | file `*.pth.json` nằm cạnh checkpoint |

Metadata ghi lại `state_dim`, `action_dim`, action bounds, actuator order,
observation layout, calibration RPM, reward, tần số điều khiển và curriculum.
Resume chỉ được chấp nhận khi sidecar tồn tại và tối thiểu khớp:

```text
policy_type = motor_level_ppo
state_dim = 17
action_dim = 4
actuator_order = [FL, FR, RL, RR]
```

Luôn lưu file `.pth` cùng file `.pth.json`; không đổi tên hoặc tráo sidecar giữa
các run.

### 10.9. Lệnh chạy và các tham số có thể cấu hình từ CLI

Chạy từ repository root:

```powershell
python -m drone_control.action_spaces.motor_residual.train --help

python -m drone_control.action_spaces.motor_residual.train `
  --max-timesteps 500000 `
  --seed 42 `
  --monitor-every 10 `
  --monitor-window 20
```

Resume từ checkpoint tương thích:

```powershell
python -m drone_control.action_spaces.motor_residual.train `
  --resume "checkpoints_motor\<run_id>\episode_001000_step_000050000.pth" `
  --max-timesteps 100000 `
  --seed 42
```

`--quiet-monitor` chỉ tắt báo cáo trên console; `training.csv` và
`training_status.json` vẫn được cập nhật. Các siêu tham số PPO còn lại hiện là
hằng số trong `motor_residual/train.py`; nếu thay đổi chúng cho một thí nghiệm,
cần lưu cấu hình thí nghiệm và không giả định checkpoint cũ còn tương thích chỉ
vì cùng `state_dim` và `action_dim`.
