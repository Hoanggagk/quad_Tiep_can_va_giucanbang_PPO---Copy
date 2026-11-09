# Hướng dẫn sử dụng PPO với MAVSDK

## Mô tả

Module `drone_control.action_spaces.velocity_ned.control_mavsdk` triển khai
điều khiển drone bằng mô hình PPO đã huấn luyện thông qua MAVSDK,
sử dụng lệnh `set_velocity_ned()` trong hệ tọa độ NED.

## Yêu cầu

### 1. Môi trường Python

```bash
pip install mavsdk numpy torch
```

### 2. PX4 SITL

- Chạy PX4 SITL với kết nối UDP: `udpin://127.0.0.1:14550`
- Hoặc kết nối với drone thực qua MAVSDK

### 3. Mô hình PPO đã huấn luyện

- Đảm bảo đã chạy module `velocity_ned.train` để tạo checkpoint
- Checkpoint mặc định: `log_dir/hover_projectairsim/ppo_hover_final.pth`

## Cấu hình

Mở file `drone_control/action_spaces/velocity_ned/control_mavsdk.py` và điều chỉnh các tham số:

```python
# Đường dẫn checkpoint PPO
CHECKPOINT_PATH = "log_dir/hover_projectairsim/ppo_hover_final.pth"

# Điểm mục tiêu hover (NED frame)
GOAL_POSITION = [0.0, 0.0, -3.0]  # x=0m, y=0m, z=-3m (3m trên mặt đất)

# Thời gian và tần số điều khiển
CONTROL_DURATION = 30  # giây
CONTROL_RATE = 10      # Hz (0.1s mỗi bước)
```

## Cách chạy

### Bước 1: Khởi động PX4 SITL

```bash
# Terminal 1: Khởi động PX4 SITL với Gazebo
cd ~/PX4-Autopilot
make px4_sitl gazebo
```

### Bước 2: Chạy điều khiển PPO

```bash
# Terminal 2: Chạy script điều khiển
python -m drone_control.action_spaces.velocity_ned.control_mavsdk
```

## Quy trình hoạt động

1. **Nạp mô hình PPO** từ checkpoint
2. **Kết nối với PX4** qua MAVSDK (UDP port 14550)
3. **Chờ GPS ready** và arm drone
4. **Khởi động Offboard mode**
5. **Vòng lặp điều khiển** (30 giây):
   - Thu thập telemetry từ drone:
     - Position NED (x, y, z)
     - Attitude Euler (roll, pitch, yaw)
     - Velocity NED (vx, vy, vz)
     - Angular velocity (roll_rate, pitch_rate, yaw_rate)
     - Distance to goal
   - Tạo observation vector 13 chiều
   - Chọn action từ PPO: `[v_north, v_east, v_down]`
   - Gửi lệnh điều khiển qua `set_velocity_ned()`
6. **Dừng Offboard** và hạ cánh
7. **Lưu trajectory** vào file CSV

## Output

### Console log

```
[PPO] Đang nạp model từ: log_dir/hover_projectairsim/ppo_hover_final.pth
[PPO] Đã nạp model thành công.
[MAVSDK] Đang kết nối PX4...
[MAVSDK] Đã kết nối với PX4
[MAVSDK] GPS OK
[CONTROL] Bắt đầu điều khiển PPO (hover tại [0.0, 0.0, -3.0])
  [  0.0s] Pos=(  0.05,   0.02,  -0.15) | RPY=(  0.5°,  -0.3°,   0.1°) | Dist= 2.85m | Cmd=( 0.12, -0.05, -0.45) m/s
  [  1.0s] Pos=( -0.08,   0.11,  -0.62) | RPY=(  0.3°,   0.1°,  -0.2°) | Dist= 2.39m | Cmd=( 0.08,  0.03, -0.38) m/s
  ...
[CONTROL] Hoàn thành 30s điều khiển.
[SAVE] Trajectory đã lưu: ppo_mavsdk_trajectory.csv
```

### File CSV trajectory

File `ppo_mavsdk_trajectory.csv` chứa:

- `time_s`: Thời gian (giây)
- `x_m, y_m, z_m`: Vị trí NED (mét)
- `roll_deg, pitch_deg, yaw_deg`: Góc Euler (độ)

## Lưu ý quan trọng

### 1. Hệ tọa độ NED

- **N** (North): X dương hướng Bắc
- **E** (East): Y dương hướng Đông
- **D** (Down): Z dương hướng xuống đất
- Ví dụ: z=-3 nghĩa là 3m trên mặt đất

### 2. Action space

- PPO output: `[v_north, v_east, v_down]` ∈ [-1.5, 1.5] m/s
- Giá trị âm: di chuyển ngược hướng (South/West/Up)
- Giá trị dương: di chuyển thuận hướng (North/East/Down)

### 3. Safety

- Kiểm tra GOAL_POSITION phù hợp với môi trường
- Điều chỉnh CONTROL_DURATION để tránh bay quá lâu
- Luôn có nút emergency stop sẵn sàng

### 4. Kết nối MAVSDK

Nếu thay đổi địa chỉ kết nối, sửa dòng:

```python
await drone.connect(system_address="udpin://127.0.0.1:14550")
```

Các kết nối phổ biến:

- SITL: `udpin://127.0.0.1:14550`
- Serial: `serial:///dev/ttyUSB0:57600`
- TCP: `tcp://192.168.1.100:5760`

## Troubleshooting

### Lỗi: "Không tìm thấy checkpoint"

- Chạy module `velocity_ned.train` trước để tạo model
- Hoặc thay đổi `CHECKPOINT_PATH` đến checkpoint có sẵn

### Lỗi: "Không lấy được telemetry"

- Kiểm tra kết nối MAVSDK với PX4
- Đảm bảo PX4 đã boot xong và telemetry hoạt động
- Kiểm tra firewall không chặn UDP port

### Lỗi: "Lỗi Offboard"

- Đảm bảo đã gửi setpoint trước khi start offboard
- Kiểm tra PX4 parameter `COM_RCL_EXCEPT` (RC loss exception)
- Xem log PX4: `dmesg -w`

### Drone không hover tốt

- Kiểm tra `GOAL_POSITION` có khớp với môi trường train không
- Thử với checkpoint khác (ep*\_step*.pth)
- Giảm `MIN_ACTION_STD` để action ổn định hơn

## So sánh với ProjectAirSim

| Khía cạnh  | ProjectAirSim (`velocity_ned.train`) | MAVSDK (`velocity_ned.control_mavsdk`) |
| ---------- | ------------------------------ | -------------------------------- |
| Môi trường | Unity simulation               | PX4 SITL/Gazebo hoặc drone thực  |
| Kết nối    | HTTP API                       | MAVSDK (gRPC)                    |
| Điều khiển | `set_velocity()` của sim       | `set_velocity_ned()` qua MAVLink |
| Telemetry  | Callback từ sim                | Async generator từ PX4           |
| Mục đích   | Huấn luyện PPO                 | Triển khai model đã train        |

## Tham khảo

- [MAVSDK Python Docs](https://mavsdk.mavlink.io/main/en/python/)
- [PX4 Offboard Control](https://docs.px4.io/main/en/flight_modes/offboard.html)
- [NED Coordinate System](https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates)
