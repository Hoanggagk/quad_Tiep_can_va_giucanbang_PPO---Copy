# Bảng Tham Số Cấu Hình Thuật Toán PPO

> Các mục 1–9 bên dưới mô tả cấu hình `velocity_ned` với 3 hành động vận tốc.
> Cấu hình PPO điều khiển trực tiếp 4 động cơ (`motor_residual`) được trình bày
> riêng tại [Mục 10](#10-cấu-hình-ppo-cho-không-gian-hành-động-4-động-cơ).
> Công thức reward đầy đủ được tổng hợp riêng tại
> [PPO_REWARD_FUNCTION.md](PPO_REWARD_FUNCTION.md).

## Cách đọc tài liệu: tại sao có hai reward function?

Reward function **không phải là một siêu tham số dùng chung của PPO**. PPO dùng
chung chỉ tối ưu tổng reward do environment trả về. Vì repository có hai
environment với hai không gian hành động khác nhau, tài liệu phải mô tả hai
reward function tương ứng:

| Cấu hình | Environment | Policy điều khiển | Reward được mô tả tại |
|---|---|---|---|
| `velocity_ned` | `action_spaces/velocity_ned/env.py` | 3 vận tốc `[v_north, v_east, v_down]` | [Mục 7](#7-reward-function-của-velocity_ned-3-action-vận-tốc) |
| `motor_residual` | `action_spaces/motor_residual/env.py` | 4 residual motor `[FL, FR, RL, RR]` | [Mục 10.6](#106-reward-function-của-motor_residual-4-action-động-cơ) |

Hai reward này không được cộng với nhau và không được chọn đồng thời trong một
lần huấn luyện. Entry point được chạy sẽ quyết định environment và reward:

```text
velocity_ned.train   → DroneHoveringEnv.get_reward(action)
motor_residual.train → DroneMotorHoverEnv.calculate_reward(...)
```

Các siêu tham số PPO như `GAMMA`, `EPS_CLIP`, `K_EPOCHS`, learning rate và kiến
trúc Actor-Critic có thể giống nhau, nhưng reward phải khác vì mức điều khiển
khác nhau. Với motor-level control, agent trực tiếp tạo lệnh cho rotor nên cần
thêm các penalty về motor effort, độ mượt và saturation mà velocity-level
control không cần xử lý trực tiếp.

## 1. Tham Số Không Gian (State & Action Space)

| Tham Số        | Ký Hiệu | Giá Trị     | Ý Nghĩa                                                                                                             |
| -------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| `STATE_DIM`    | $N_s$   | 13          | Chiều không gian trạng thái: [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal] |
| `ACTION_DIM`   | $N_a$   | 3           | Chiều không gian hành động: [v_north, v_east, v_down]                                                               |
| `Action Range` | $A$     | [-1.5, 1.5] m/s | Velocity setpoint NED theo từng trục |

---

## 2. Tham Số Độ Lệch Chuẩn Hành Động (Action Standard Deviation)

| Tham Số                 | Ký Hiệu        | Giá Trị     | Ý Nghĩa                                                             |
| ----------------------- | -------------- | ----------- | ------------------------------------------------------------------- |
| `ACTION_STD_INIT`       | $\sigma_0$     | 1.5         | Profile chính `velocity_ned.train`; short-episode là profile riêng |
| `MIN_ACTION_STD`        | $\sigma_{min}$ | 0.01        | Độ lệch chuẩn tối thiểu (~1.7% của range, ≈±0.15 m/s khi khai thác) |
| `ACTION_STD_DECAY_RATE` | $\beta$        | 0.01        | Tốc độ giảm std mỗi lần giảm                                        |
| `ACTION_STD_DECAY_FREQ` | $F_\sigma$     | 50 episodes | Tần suất giảm action_std (cứ 50 episode)                            |

**Giải thích**: Độ lệch chuẩn cao ban đầu giúp khám phá (exploration), sau đó giảm dần để khai thác (exploitation).

---

## 3. Tham Số Vòng Lặp Huấn Luyện (Training Loop)

| Tham Số                  | Ký Hiệu   | Giá Trị         | Ý Nghĩa                                              |
| ------------------------ | --------- | --------------- | ---------------------------------------------------- |
| `MAX_TRAINING_TIMESTEPS` | $T_{max}$ | 5×10⁵ (500,000) | Tổng số timestep tối đa                              |
| `STEPS_PER_EPISODE`      | $S_e$     | 150             | Số bước tối đa mỗi episode (150 × 0.1s = 15s mô phỏng) |
| `UPDATE_TIMESTEP`        | $T_{up}$  | 2,000           | Cập nhật PPO sau mỗi N timestep                      |
| `K_EPOCHS`               | $K$       | 30              | Số epoch per update (lặp trên dữ liệu trong 1 batch) |

**Giải thích**:

- Mỗi episode chịu tối đa 150 bước
- Sau mỗi 2,000 timestep, cập nhật network PPO
- Mỗi lần update, lặp 30 lần trên dữ liệu batch

---

## 4. Tham Số Thuật Toán PPO (PPO Algorithm)

| Tham Số     | Ký Hiệu    | Giá Trị         | Ý Nghĩa                                                |
| ----------- | ---------- | --------------- | ------------------------------------------------------ |
| `EPS_CLIP`  | $\epsilon$ | 0.2             | Clip hệ số: π_new / π_old ∈ [1-ε, 1+ε] = [0.8, 1.2]    |
| `GAMMA`     | $\gamma$   | 0.99            | Hệ số chiết khấu (discount factor) - giá trị tương lai |
| `LR_ACTOR`  | $\alpha_a$ | 3×10⁻⁴ (0.0003) | Learning rate cho Actor network                        |
| `LR_CRITIC` | $\alpha_c$ | 1×10⁻³ (0.001)  | Learning rate cho Critic network                       |
| `GAE_LAMBDA` | $\lambda$ | 0.95 | GAE; bootstrap ở truncation, không bootstrap terminal vật lý |
| `TARGET_KL` | — | 0.02 | Dừng sớm epoch PPO khi approximate KL vượt ngưỡng |

**Giải thích**:

- `EPS_CLIP = 0.2`: Hạn chế mức độ thay đổi policy trong mỗi update (tránh thay đổi quá tột)
- `GAMMA = 0.99`: Cân bằng giữa reward hiện tại và tương lai
- `LR_CRITIC > LR_ACTOR`: Critic học nhanh hơn để ước lượng value chính xác

---

## 5. Tham Số Kiểm Soát Log (Logging & Checkpointing)

| Tham Số           | Ký Hiệu | Giá Trị               | Ý Nghĩa                             |
| ----------------- | ------- | --------------------- | ----------------------------------- |
| `PRINT_FREQ`      | $F_p$   | 3,000 timesteps       | In log console mỗi N timestep       |
| `LOG_FREQ`        | $F_l$   | 750 timesteps         | Ghi CSV log mỗi N timestep          |
| `SAVE_MODEL_FREQ` | $F_s$   | 50,000 timesteps      | Lưu checkpoint mỗi N timestep       |
| `LOG_DIR`         | —       | "log_dir/"            | Thư mục lưu trữ logs và checkpoints |
| `RUN_NAME`        | —       | "hover_projectairsim" | Tên run (tạo subfolder)             |

---

## 6. Tham Số Môi trường (Environment)

| Tham Số                 | Ký Hiệu      | Giá Trị     | Ý Nghĩa                                   |
| ----------------------- | ------------ | ----------- | ----------------------------------------- |
| `dt`                    | $\Delta t$   | 0.1 seconds | Thời gian giữa các lần gửi lệnh hành động |
| `MAX_STEPS_PER_EPISODE` | $S_{max}$    | 150 bước    | = 150 × 0.1s = 15s mô phỏng               |
| `Goal Pose`             | $\mathbf{g}$ | [0, 0, -2]  | Mục tiêu bay: NED (North, East, Down)     |

---

## 7. Reward function của `velocity_ned` (3 action vận tốc)

> Phần này **chỉ áp dụng cho các lệnh chạy
> `drone_control.action_spaces.velocity_ned.*`**. Không dùng các trọng số này để
> diễn giải log của `motor_residual`.

| Tham Số             | Ký Hiệu    | Giá Trị | Ý Nghĩa                                                  |
| ------------------- | ---------- | ------- | -------------------------------------------------------- |
| `r_pos` weight      | $w_p$      | 1.0     | Thành phần vị trí (1.0 khi ở tâm, giảm đến 0 nếu quá xa) |
| `r_approach` weight | $w_a$      | 2.0     | Thưởng khi tiếp cận mục tiêu                             |
| `r_att` weight      | $w_{att}$  | -0.1    | Phạt nếu drone nghiêng quá nhiều                         |
| `survival bonus`    | $r_s$      | 0.01    | Thưởng sống sót mỗi bước                                 |
| `hover bonus`       | $r_h$      | 0.5     | Thưởng khi đồng thời đạt ngưỡng vị trí, vận tốc, attitude và yaw |
| `success bonus`     | $r_{succ}$ | 5.0     | Chỉ phát một lần sau 2 s hover liên tục rồi terminate success |
| `crash penalty`     | $r_c$      | -2.0    | Phạt khi va chạm hoặc quá xa                             |
| `ground penalty`    | $r_g$      | -2.0    | Phạt khi chạm mặt đất                                    |

### 7.1. Công thức từng thành phần

Ký hiệu tại bước $t$:

- $\mathbf{p}_t$ là vị trí drone, $\mathbf{g}$ là goal và
  $d_t=\lVert\mathbf{p}_t-\mathbf{g}\rVert_2$;
- $\phi_t,\theta_t$ lần lượt là roll và pitch;
- $\mathbb{1}[\cdot]$ bằng 1 khi điều kiện đúng, ngược lại bằng 0.

Các thành phần được tính như sau:

$$
r_{pos,t}=\max\left(0,1-\frac{d_t}{3}\right)
$$

$$
r_{approach,t}=
\begin{cases}
2(d_{t-1}-d_t), & \text{khi đã có state trước đó},\\
0, & \text{ở bước đầu tiên}.
\end{cases}
$$

$$
r_{att,t}=-0.1\left(|\phi_t|+|\theta_t|\right),
\qquad r_{vel,t}=0,
\qquad r_{alive,t}=0.01.
$$

Với state machine dùng chung:

$$
r_{hover,t}=0.5\,\mathbb{1}[\text{sustaining}_t],
\qquad
r_{success,t}=5\,\mathbb{1}[\text{success\_event}_t].
$$

`sustaining` yêu cầu đồng thời $d_t<0.20$ m,
$\lVert\mathbf{v}_t\rVert_2<0.20$ m/s,
$\sqrt{\phi_t^2+\theta_t^2}<10^\circ$ và sai lệch yaw đã wrap nhỏ hơn
$15^\circ$. `success_event` chỉ xuất hiện một lần sau 2 s `sustaining` liên
tục, tương đương 20 bước với $dt=0.1$ s.

### 7.2. Tổng reward

Nếu environment đã đánh dấu `crashed`, hàm trả ngay:

$$
R_t=-2.
$$

Ngược lại:

$$
R_t=r_{pos,t}+r_{approach,t}+r_{vel,t}+r_{att,t}
    +r_{alive,t}+r_{hover,t}+r_{success,t}.
$$

Pose timeout ném `TimeoutError` trước khi trả transition nên không tạo một
reward $-2$ thông thường.

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

`STATE_DIM = 13` và `observation_schema_version = 3`. Policy chỉ sử dụng các
phần tử từ chỉ số 0 đến 12. Thứ tự phải được giữ nguyên khi huấn luyện,
resume hoặc triển khai policy:

| Chỉ số | Tên trong metadata | Đại lượng |
|---:|---|---|
| 0–2 | `x_m`, `y_m`, `z_m` | Vị trí NED, đơn vị m |
| 3–5 | `roll_rad`, `pitch_rad`, `yaw_rad` | Góc Euler, đơn vị rad |
| 6–8 | `vx_mps`, `vy_mps`, `vz_mps` | Vận tốc tuyến tính, đơn vị m/s |
| 9–11 | `roll_rate_radps`, `pitch_rate_radps`, `yaw_rate_radps` | Vận tốc góc, đơn vị rad/s |
| 12 | `distance_to_goal_m` | Khoảng cách Euclid tới mục tiêu, đơn vị m |

`LayerNorm(13)` được áp dụng trước Actor và Critic. State không chứa motor
signal hoặc vector `[dx,dy,dz]`; vì vậy curriculum mặc định giữ goal cố định
tại `[0,0,-2]` và tạo bài toán reaching bằng cách thay đổi trạng thái khởi tạo.
Nếu cần goal thay đổi theo episode, phải bổ sung lại vector hướng tới goal và
tăng observation schema. Checkpoint state 20 trước đây không còn tương thích.

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
h = hover_control_signal + hover_signal_bias

nếu a_i < 0: control_signal_i = h * (1 + a_i)
nếu a_i ≥ 0: control_signal_i = h + a_i * (1 - h)

rpm_i = max_rpm * sqrt(control_signal_i)
```

Ánh xạ hover-centered sử dụng toàn bộ miền `control_signal ∈ [0, 1]` của rotor
ProjectAirSim: `a_i=-1` dừng rotor, `a_i=0` là operating point hover và
`a_i=1` là `max_rpm`. Environment không clip action; giá trị ngoài `[-1,1]`
được xem là lỗi. Policy motor dùng `tanh-squashed Gaussian` để action sinh ra
luôn nằm trong miền hợp lệ. Cần hiệu chuẩn lại khi đổi model động cơ, khối
lượng, thrust model hoặc scene vật lý.

### 10.4. Siêu tham số PPO mặc định

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| `ACTION_STD_INIT` | 0.50 | Tăng độ lệch chuẩn khám phá ban đầu trên latent action |
| `ACTION_STD_DECAY_RATE` | 0.005 | Giảm chậm để duy trì exploration lâu hơn |
| `ACTION_STD_DECAY_FREQ_EPISODES` | 100 episode | Giãn chu kỳ giảm độ lệch chuẩn |
| `MIN_ACTION_STD` | 0.10 | Duy trì mức exploration tối thiểu cao hơn |
| `MAX_TRAINING_TIMESTEPS` | 500,000 | Tổng số bước huấn luyện mặc định |
| `STEPS_PER_EPISODE` | 500 | Horizon tối đa của một episode |
| `UPDATE_TIMESTEP` | 2,000 | Số transition giữa hai lần PPO update |
| `K_EPOCHS` | 30 | Số epoch tối ưu trên mỗi rollout batch |
| `EPS_CLIP` | 0.2 | Ngưỡng PPO clipped objective |
| `GAMMA` | 0.99 | Hệ số chiết khấu |
| `LR_ACTOR` | 3×10⁻⁴ | Learning rate của Actor |
| `LR_CRITIC` | 1×10⁻³ | Learning rate của Critic |
| `GAE_LAMBDA` | 0.95 | GAE có bootstrap đúng ở time-limit truncation |
| `TARGET_KL` | 0.02 | Early-stop epoch nếu approximate KL vượt ngưỡng |
| `SAVE_MODEL_FREQ` | 50,000 timestep | Chu kỳ lưu checkpoint |

Với `CONTROL_FREQUENCY_HZ = 50`, ta có `dt = 0.02 s`; horizon 500 bước tương
đương tối đa 10 giây mô phỏng. Một PPO update nhận tối đa khoảng bốn episode
đủ horizon, nhưng số episode thực tế có thể lớn hơn do termination sớm.

### 10.5. Kiến trúc Actor-Critic

```text
State 13D → LayerNorm
  ├─ Actor mean: 13 → 64 → Tanh → 64 → Tanh → 4 (latent mean)
  │             z ~ Normal(mean, std), action = tanh(z)
  └─ Critic: 13 → 64 → Tanh → 64 → Tanh → 1
```

Actor motor sử dụng `tanh-squashed Gaussian` có hiệu chỉnh Jacobian trong
`log_prob`, nên action nằm trong `(-1,1)^4` mà không cần clamp. Cấu hình
`velocity_ned` vẫn giữ distribution cũ để bảo toàn checkpoint và hành vi hiện
tại của pipeline vận tốc.

### 10.6. Reward function của `motor_residual` (4 action động cơ)

> Phần này **chỉ áp dụng cho
> `drone_control.action_spaces.motor_residual.train`**. Từ `REWARD_VERSION = 2`,
> công thức shaping được đồng nhất với `velocity_ned`; khác biệt chỉ còn ở
> action space, observation và tần số điều khiển.

Ký hiệu $d_t=\lVert\mathbf{p}_t-\mathbf{g}\rVert_2$, và
$\phi_t,\theta_t$ là roll, pitch. Các thành phần reward theo từng bước là:

$$
r_{pos,t}=\max\left(0,1-\frac{d_t}{3}\right),
$$

$$
r_{approach,t}=
\begin{cases}
2(d_{t-1}-d_t), & \text{khi đã có state trước đó},\\
0, & \text{ở bước đầu tiên}.
\end{cases}
$$

$$
r_{att,t}=-0.1(|\phi_t|+|\theta_t|),
\qquad r_{vel,t}=0,
\qquad r_{alive,t}=0.01.
$$

`SuccessTracker` dùng chung quyết định hover và success:

$$
r_{hover,t}=0.5\,\mathbb{1}[\text{sustaining}_t],
\qquad
r_{success,t}=5\,\mathbb{1}[\text{success\_event}_t].
$$

Nếu `crashed` là true, reward trả về đúng $-2$. Ngược lại:

$$
R_t=r_{pos,t}+r_{approach,t}+r_{vel,t}+r_{att,t}
    +r_{alive,t}+r_{hover,t}+r_{success,t}.
$$

Reward không còn scale theo `dt` và không còn các penalty `yaw`,
`angular_rate`, `motor_effort`, `motor_smoothness` hay `saturation`. Các trường
`position_error_m`, `velocity_norm_mps`, `angular_rate_norm_radps` và
`saturation_fraction` vẫn được ghi trong `reward_terms` để giám sát; chúng
không được cộng vào $R_t$.

### 10.7. Termination và curriculum

Episode kết thúc sớm khi xảy ra một trong các điều kiện:

- collision;
- `|roll|` hoặc `|pitch|` vượt giới hạn attitude của environment;
- chạm gần mặt đất (`z > -0.2 m` trong hệ NED);
- khoảng cách tới mục tiêu lớn hơn `5 m`;
- chuẩn vận tốc lớn hơn `5 m/s`.

Curriculum mặc định tiến qua tám stage:

```text
attitude_hover → random_attitude → random_position_velocity
→ altitude_change → takeoff → target_reaching → disturbance
→ domain_randomization
```

Mọi stage mặc định dùng cùng goal `[0,0,-2]`. Các stage vẫn tăng độ khó bằng
initial position, attitude, velocity, disturbance và domain randomization.

Stage chỉ được nâng khi đủ số episode đánh giá và đồng thời đạt các ngưỡng về
survival rate, reach/sustained success rate, position RMSE và attitude RMSE. Khi resume, stage hiện tại được
khôi phục từ metadata của checkpoint.

### 10.8. Logging, checkpoint và tương thích

Mỗi lần chạy tạo `run_id` riêng:

| Artifact | Vị trí |
|---|---|
| Log từng episode | `log_dir/hover_motor/<run_id>/training.csv` |
| Trạng thái monitor mới nhất | `log_dir/hover_motor/<run_id>/training_status.json` |
| Checkpoint | `checkpoints_motor/<run_id>/*.pth` |
| Metadata checkpoint | file `*.pth.json` nằm cạnh checkpoint |

Metadata schema v4 ghi lại `state_dim`, `action_dim`, action bounds, actuator order,
observation schema/layout, task contract v1, reward version, PPO/GAE config,
action-std schedule, calibration RPM, tần số điều khiển và curriculum.
Resume chỉ được chấp nhận khi sidecar tồn tại và tối thiểu khớp:

```text
policy_type = motor_level_ppo
schema_version = 4
state_dim = 13
action_dim = 4
observation_schema_version = 3
actuator_order = [FL, FR, RL, RR]
action_mapping = hover_centered_full_signal
action_mapping_version = 2
policy_distribution = tanh_squashed_gaussian
```

Checkpoint motor tạo trước mapping version 2 không được resume vì cùng một
action nay tạo control signal khác. Luôn lưu file `.pth` cùng file `.pth.json`;
không đổi tên hoặc tráo sidecar giữa các run.

Checkpoint state 17 hoặc state 20 trước đây không được resume vào policy state
13. Chúng chỉ là artifact lịch sử; so sánh kết quả phải ghi rõ observation
schema tương ứng.

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

### 10.10. Manifest đánh giá độc lập

Tạo manifest frozen cho Test A–D mà không khởi động simulator:

```powershell
python tools/create_evaluation_manifest.py `
  --action-space motor_residual `
  --checkpoint "checkpoints_motor\<run_id>\final.pth" `
  --seeds 11 22 33 44 55 `
  --output "evaluation_results\motor_manifest.json"
```

Tool từ chối overwrite manifest hiện có. Protocol và bộ metric nằm trong
`drone_control/evaluation_protocol.py`; chạy simulator/evaluation thật vẫn là
một bước có chủ đích và không được tự động kích hoạt bởi lệnh tạo manifest.
