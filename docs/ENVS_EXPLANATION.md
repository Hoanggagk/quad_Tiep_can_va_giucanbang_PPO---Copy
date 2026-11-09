# Giải thích NED velocity environment

## Tổng quan

`drone_control/action_spaces/velocity_ned/env.py` định nghĩa lớp
`DroneHoveringEnv` - một môi trường mô phỏng drone hover sử dụng
ProjectAirSim. Đây là môi trường theo chuẩn Gymnasium dùng để huấn
luyện agent RL bằng PPO.

---

## Kiến trúc lớp

### Lớp: `DroneHoveringEnv(ProjectAirSimEnv)`

Kế thừa từ `ProjectAirSimEnv` của thư viện ProjectAirSim để mô phỏng drone.

---

## Các hằng số và cấu hình chính

```python
MAX_STEPS_PER_EPISODE = 50      # Tối đa 50 bước = 50 × 0.1s = 5 giây thực
self.dt = 0.1                   # Thời gian giữa các bước: 0.1 giây
self.sim_config_fname = "scene_basic_drone.jsonc"  # File cấu hình scene
```

### Không gian hành động (Action Space)

- **Loại**: Box liên tục (Continuous)
- **Hình dạng**: (3,) - 3 chiều vận tốc
- **Phạm vi**: [-1.5, 1.5]
- **Ý nghĩa**:
  - `action[0]`: vận tốc hướng Bắc (v_north)
  - `action[1]`: vận tốc hướng Đông (v_east)
  - `action[2]`: vận tốc hướng Xuống (v_down)

### Không gian quan sát (Observation Space)

- **Loại**: Box liên tục
- **Hình dạng**: (13,)
- **Phạm vi**: [-∞, +∞]
- **Thành phần**:
  1. `x, y, z` (3 chiều): vị trí drone trong hệ tọa độ NED
  2. `roll, pitch, yaw` (3 chiều): tư thế (góc quay)
  3. `vx, vy, vz` (3 chiều): vận tốc tuyến tính
  4. `roll_rate, pitch_rate, yaw_rate` (3 chiều): vận tốc góc
  5. `dist_to_goal` (1 chiều): khoảng cách đến mục tiêu

---

## Các phương thức chính

### `__init__()`

Khởi tạo môi trường:

- Tải cấu hình scene
- Định nghĩa không gian hành động và quan sát
- Khởi tạo trạng thái mặc định
- Mục tiêu: bay tới vị trí [0, 0, -2] (2m cao so với vị trí xuất phát)

### `reset(seed=None, options=None)`

**Mục đích**: Khởi tạo lại môi trường ở đầu mỗi episode

- Hủy subscriptions cũ (tránh memory leak)
- Load lại scene configuration
- Khởi tạo drone mới
- Đặt mục tiêu bay: [x=0, y=0, z=-2]
- Subscribe đến topic pose để nhận cập nhật vị trí
- Arm drone (khởi động)
- Tạm dừng mô phỏng (pause) để manual stepping
- **Trả về**: quan sát ban đầu, metadata rỗng

### `step(action)` (Async)

**Mục đích**: Thực hiện một bước mô phỏng

```
1. Xử lý hành động: clip vào [-1.5, 1.5]
2. Gọi move_by_velocity_async() để drone di chuyển
3. Lock-step simulation: đặt mục tiêu timestamp
4. Continue sim không blocking, spin đợi pose sync
5. Lấy quan sát tiếp theo
6. Kiểm tra điều kiện kết thúc
7. Tính reward
8. Trả về: (next_obs, reward, terminated, truncated, info)
```

**Timeout**: 5 giây (50 bước × 0.1s)

### `get_observation()`

**Mục đích**: Tạo vector quan sát 13 chiều

- Nếu chưa nhận pose từ callback, lấy trực tiếp từ sim
- Tính khoảng cách đến mục tiêu
- Trả về observation dưới dạng numpy array float32

### `pose_callback(pose)`

**Mục đích**: Cập nhật vị trí, tư thế và ước lượng vận tốc (gọi bất cứ khi nào pose update)

- **Xử lý**:
  - Lấy timestamp từ sim (nanosecond)
  - Lấy vị trí (position) và tư thế (orientation - quaternion)
  - Chuyển quaternion → roll/pitch/yaw bằng `quaternion_to_rpy()`
  - Ước lượng vận tốc bằng vi phân hữu hạn: `v = Δpos / Δt`
  - Xử lý wraparound góc yaw (±π) để tránh spike vận tốc góc
  - Đánh dấu pose đã sync nếu timestamp ≥ target_simtime (lock-step)

### `_check_termination()`

**Mục đích**: Kiểm tra điều kiện kết thúc sớm
4 điều kiện kết thúc:

1. **Drone lật**: |roll| > 90° hoặc |pitch| > 90°
2. **Chạm đất**: z > -0.2m (z=0 là mặt đất)
3. **Quá xa mục tiêu**: khoảng cách > 5m
4. **Vận tốc nguy hiểm**: ||v|| > 5 m/s

### `get_reward(action)`

**Mục đích**: Tính reward cho bước hiện tại

**Các thành phần reward**:

| Thành phần   | Công thức                            | Ý nghĩa                           |
| ------------ | ------------------------------------ | --------------------------------- |
| `r_pos`      | `max(0, 1.0 - dist/3.0)`             | Thưởng khi gần mục tiêu           |
| `r_approach` | `(prev_err - curr_err) × 2.0`        | Thưởng khi lại gần                |
| `r_vel`      | `-0.5 × \|\|v\|\|` (nếu dist < 0.6m) | Phạt khi chuyển động gần mục tiêu |
| `r_att`      | `-0.1 × (\|roll\| + \|pitch\|)`      | Phạt khi lệch tư thế              |
| Survival     | `0.01`                               | Thưởng tồn tại                    |
| Hover bonus  | `0.5` (+ 5.0 nếu > 20 bước)          | Thưởng khi bay lơ lửng ổn định    |

**Điều kiện thành công**:

- Khi `dist < 0.2m` và `\|\|v\|\| < 0.2 m/s` liên tục > 20 bước:
  - Thêm 5.0 reward
  - Đặt `done = True` (episode kết thúc)

---

## Cơ chế Lock-Step Simulation

Mục đích: **Đồng bộ hóa mô phỏng với RL training** để tránh skip lệnh.

### Quy trình:

1. **step()** gửi lệnh `move_by_velocity_async()` cho 0.1s
2. **step()** tính timestamp mục tiêu: `t_next = current_time + 0.1s (nanosec)`
3. **step()** gọi `continue_for_sim_time()` (NON-BLOCKING)
4. **step()** spin async loop: đợi `pose_sync = True`
5. **pose_callback()** kiểm tra: nếu `sim_time ≥ t_target` thì set `_pose_sync = True`
6. **step()** tiếp tục khi pose đã sync

✓ Đảm bảo mỗi hành động được sim chính xác đủ lâu

---

## Các biến trạng thái quan trọng

| Biến               | Kiểu                             | Mục đích                             |
| ------------------ | -------------------------------- | ------------------------------------ |
| `actor_pose`       | list [x, y, z, roll, pitch, yaw] | Vị trí/tư thế hiện tại               |
| `linear_velocity`  | tuple (vx, vy, vz)               | Vận tốc tuyến tính                   |
| `angular_velocity` | tuple (ωx, ωy, ωz)               | Vận tốc góc                          |
| `done`             | bool                             | Kết thúc episode (success/failure)   |
| `crashed`          | bool                             | Drone bị crash                       |
| `ground_touch`     | bool                             | Chạm đất                             |
| `_hover_steps`     | int                              | Số bước liên tục bay lơ lửng ổn định |
| `_step_count`      | int                              | Số bước hiện tại                     |
| `_target_simtime`  | int (nanosec)                    | Timestamp mục tiêu cho lock-step     |
| `_pose_sync`       | bool                             | Flag cờ đã sync pose với target time |
| `prev_state`       | ndarray                          | Quan sát bước trước                  |
| `prev_action`      | ndarray                          | Hành động bước trước                 |

---

## Thread Safety

Sử dụng `threading.Lock` (`self._pose_lock`) để bảo vệ các biến trạng thái:

- `actor_pose`
- `linear_velocity`
- `angular_velocity`

Tránh race condition giữa main thread (RL) và callback thread (sim updates)

---

## Luồng hoạt động của một Episode

```
1. reset()
   ├─ Load scene & drone
   ├─ Subscribe pose callback
   ├─ Arm drone
   └─ Trả về obs

2. Lặp 50 bước:
   ├─ step(action)
   │  ├─ Gửi lệnh vận tốc
   │  ├─ Continue sim (lock-step)
   │  ├─ Spin đợi pose sync
   │  ├─ get_observation()
   │  ├─ _check_termination()
   │  ├─ get_reward()
   │  └─ Trả về (obs, reward, done, truncated, info)
   │
   ├─ while pose_sync & not done & step < 50:
   │  └─ Tiếp tục lặp
   │
   └─ Episode kết thúc (success/crash/timeout)

3. Episode kế tiếp: quay lại reset()
```

---

## Ghi chú Kỹ thuật

### NED Coordinate System

- **N** (North): Bắc (x dương)
- **E** (East): Đông (y dương)
- **D** (Down): Xuống (z dương = lên cao trong hệ NED âm)

→ Mục tiêu [0, 0, -2] = tại gốc, 2m cao lên

### Thời gian Simulasi

- Sử dụng **sim timestamp** (nanosecond) thay wall-clock
- Đảm bảo tính chính xác khi sim chạy nhanh/chậm hơn real-time

### Process Callback Async

- Callback chạy trên thread khác → cần lock
- Spin trong async loop với `await asyncio.sleep(0.001)`
- Tránh blocking training loop

---

## Tích hợp với PPO Training

Environment này được sử dụng cùng với
`drone_control/algorithms/ppo.py` và
`drone_control/action_spaces/velocity_ned/train.py`:

- PPO agent gọi `step(action)` ở mỗi bước
- Agent học tối ưu hóa reward
- Mục tiêu: bay lơ lửng tại [0, 0, -2] ổn định
