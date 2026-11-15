# Mô tả hàm phần thưởng của thuật toán PPO

## 1. Phạm vi

Repository có một implementation PPO dùng chung nhưng hai environment khác
nhau. Mỗi lần huấn luyện chỉ sử dụng reward của environment được chọn:

| Cấu hình | Không gian hành động | Hàm reward trong source | Reward version |
|---|---|---|---:|
| `velocity_ned` | 3 velocity setpoint NED | `DroneHoveringEnv.get_reward()` | 2 |
| `motor_residual` | 4 lệnh residual motor `[FL, FR, RL, RR]` | `DroneMotorHoverEnv.calculate_reward()` | 1 |

Hai reward function **không được cộng với nhau**. PPO chỉ nhận scalar reward
$R_t$ từ environment và tối ưu discounted return:

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k}.
$$

Các siêu tham số như $\gamma$, GAE-$\lambda$ và PPO clipping quyết định cách
PPO sử dụng chuỗi reward; chúng không thay đổi định nghĩa từng thành phần
$R_t$ bên dưới.

## 2. Ký hiệu chung

| Ký hiệu | Ý nghĩa |
|---|---|
| $t$ | Chỉ số control step hiện tại |
| $\mathbf{p}_t$ | Vị trí drone trong hệ NED, đơn vị m |
| $\mathbf{g}$ | Vị trí goal trong hệ NED, đơn vị m |
| $d_t=\lVert\mathbf{p}_t-\mathbf{g}\rVert_2$ | Khoảng cách Euclid tới goal |
| $\mathbf{v}_t$ | Vận tốc tuyến tính, đơn vị m/s |
| $\boldsymbol{\omega}_t$ | Vận tốc góc, đơn vị rad/s |
| $\phi_t,\theta_t,\psi_t$ | Roll, pitch và yaw, đơn vị rad |
| $\mathbb{1}[C]$ | Bằng 1 nếu điều kiện $C$ đúng, ngược lại bằng 0 |

Trong hệ NED, trục $z$ dương hướng xuống; goal mặc định là
$\mathbf{g}=[0,0,-2]$ m.

## 3. Reward của `velocity_ned`

### 3.1. Thành phần liên tục

Position reward giảm tuyến tính theo khoảng cách và bằng 0 từ 3 m trở lên:

$$
r_{pos,t}=\max\left(0,1-\frac{d_t}{3}\right).
$$

Approach reward đo mức giảm khoảng cách giữa hai bước:

$$
r_{approach,t}=
\begin{cases}
2(d_{t-1}-d_t), & \text{khi đã có state trước đó},\\
0, & \text{ở bước đầu tiên của episode}.
\end{cases}
$$

Do đó agent được thưởng khi $d_t<d_{t-1}$ và bị phạt khi đi xa goal.

Attitude penalty là:

$$
r_{att,t}=-0.1\left(|\phi_t|+|\theta_t|\right).
$$

Pipeline hiện tại không có velocity penalty riêng:

$$
r_{vel,t}=0.
$$

Survival reward ở mỗi bước hợp lệ là:

$$
r_{alive,t}=0.01.
$$

### 3.2. Hover và success event

`SuccessTracker` đặt `sustaining = true` khi đồng thời thỏa mãn:

$$
d_t<0.20\ \text{m},
$$

$$
\lVert\mathbf{v}_t\rVert_2<0.20\ \text{m/s},
$$

$$
\sqrt{\phi_t^2+\theta_t^2}<10^\circ,
$$

và sai lệch yaw đã wrap nhỏ hơn $15^\circ$. Khi đó:

$$
r_{hover,t}=0.5\,\mathbb{1}[\text{sustaining}_t].
$$

Nếu các điều kiện trên được duy trì liên tục trong 2 s, tương đương 20 bước
với $dt=0.1$ s, state machine chỉ phát một `success_event`:

$$
r_{success,t}=5\,\mathbb{1}[\text{success\_event}_t].
$$

Episode kết thúc ngay sau success event, vì vậy bonus 5.0 không bị cộng lặp.

### 3.3. Tổng reward và failure

Khi không crash:

$$
R_t=r_{pos,t}+r_{approach,t}+r_{vel,t}+r_{att,t}
    +r_{alive,t}+r_{hover,t}+r_{success,t}.
$$

Nếu environment đã đánh dấu `crashed`, hàm bỏ qua các thành phần trên và trả
ngay:

$$
R_t=-2.
$$

`crashed` bao gồm collision, ground touch, attitude limit, position limit và
velocity limit. Pose timeout ném `TimeoutError` trước khi trả transition nên
không tạo một reward $-2$ thông thường.

### 3.4. Đặc điểm cần lưu ý

- Action velocity không bị phạt trực tiếp trong reward.
- $r_{approach,t}$ phụ thuộc state trước đó và bằng 0 ở bước đầu episode.
- Success được xác định bằng state machine, không suy ra từ tổng reward.

## 4. Reward của `motor_residual`

Từ `REWARD_VERSION = 2`, `motor_residual` dùng cùng công thức shaping ở Mục 3:

$$
r_{pos,t}=\max\left(0,1-\frac{d_t}{3}\right),
\qquad
r_{approach,t}=2(d_{t-1}-d_t),
$$

$$
r_{vel,t}=0,
\qquad
r_{att,t}=-0.1(|\phi_t|+|\theta_t|),
\qquad
r_{alive,t}=0.01.
$$

Ở bước đầu tiên, `approach = 0`. `hover_bonus = 0.5` khi
`SuccessTracker.sustaining`; `success_bonus = 5.0` chỉ xuất hiện một lần tại
`success_event`. Nếu environment đã đánh dấu `crashed`, hàm trả đúng $-2$.
Reward được tính theo bước, không scale theo `dt`.

Các penalty motor cũ (`motor_effort`, `motor_smoothness`, `saturation`) cùng
penalty yaw và angular rate đã được bỏ khỏi tổng reward. Environment vẫn log
`position_error_m`, `velocity_norm_mps`, `angular_rate_norm_radps` và
`saturation_fraction` để đánh giá hành vi motor độc lập với shaping.

## 5. So sánh hai reward function

| Đặc điểm | `velocity_ned` | `motor_residual` |
|---|---|---|
| Mức điều khiển | Velocity setpoint | Control signal của từng rotor |
| Position shaping | Tuyến tính có chặn tại 0 | Giống `velocity_ned` |
| Approach reward | Có | Có |
| Velocity penalty | Không | Không |
| Motor effort/smoothness | Không | Không; chỉ log saturation fraction |
| Hover bonus | Theo `SuccessTracker.sustaining` | Giống `velocity_ned` |
| Success bonus | 5.0, chỉ một lần | 5.0, chỉ một lần |
| Crash handling | Trả ngay $-2$ | Trả ngay $-2$ |
| Scale theo control period | Không | Không |

## 6. Nguồn chuẩn và quy tắc cập nhật

Source code là nguồn chuẩn cuối cùng:

- `drone_control/action_spaces/velocity_ned/env.py`;
- `drone_control/action_spaces/motor_residual/env.py`;
- `drone_control/task_contract.py`;
- `drone_control/action_spaces/motor_residual/train.py` đối với reward version
  trong checkpoint metadata.

Khi thay đổi công thức hoặc trọng số reward, cần đồng thời:

1. tăng `REWARD_VERSION` nếu semantics thay đổi;
2. cập nhật unit test reward;
3. cập nhật tài liệu này và `PPO_CONFIG_TABLE.md`;
4. ghi reward version trong checkpoint/evaluation manifest;
5. không so sánh trực tiếp các run dùng reward version khác nhau.
