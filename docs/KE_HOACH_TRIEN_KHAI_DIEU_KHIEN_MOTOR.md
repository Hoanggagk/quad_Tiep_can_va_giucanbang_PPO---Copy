# Kế hoạch chuyển đổi sang điều khiển tốc độ từng động cơ

## 1. Mục tiêu

Chuyển không gian hành động của PPO từ ba lệnh vận tốc:

```text
[v_north, v_east, v_down]
```

sang bốn lệnh điều khiển rotor độc lập:

```text
[motor_FL, motor_FR, motor_RL, motor_RR]
```

Việc triển khai phải giữ nguyên workflow điều khiển vận tốc hiện tại để có thể đối chiếu và quay lại khi cần. Không ghi đè checkpoint hoặc kết quả thí nghiệm cũ.

## 2. Kiến trúc đích

```text
PPO actor
  -> action chuẩn hóa [-1, 1]^4
  -> ánh xạ quanh tốc độ hover
  -> RPM từng rotor
  -> control signal [0, 1]^4
  -> Drone.set_control_signals()
  -> 4 rotor của ProjectAirSim
```

ProjectAirSim nhận `control signal` tỷ lệ với lực đẩy, không nhận RPM tuyến tính. Quan hệ chuyển đổi là:

```text
rpm_i = max_rpm * sqrt(u_i)
u_i   = (rpm_i / max_rpm)^2
```

Trong đó `u_i` là control signal gửi tới actuator.

## 3. Nguyên tắc triển khai

- Tạo environment và cấu hình motor-level mới; không sửa trực tiếp workflow velocity-level trong giai đoạn đầu.
- Không chạy training hoặc kết nối simulator cho đến khi được yêu cầu rõ ràng.
- Không nạp checkpoint velocity-level vào mạng motor-level như một checkpoint hoàn chỉnh.
- Mọi lệnh motor phải được giới hạn trong miền an toàn `[0, 1]`.
- Khi reset, đóng environment hoặc gặp exception, phải gửi lệnh dừng cả bốn rotor.
- Sau mỗi thay đổi source code, chạy `codegraph sync .`.

## 4. Thứ tự triển khai

### Giai đoạn 1: Tạo cấu hình robot motor-level

Tạo hai file mới:

```text
sim_config/scene_basic_drone_motor.jsonc
sim_config/robot_quadrotor_motor.jsonc
```

Trong cấu hình robot, chuyển controller từ:

```jsonc
"type": "simple-flight-api"
```

sang:

```jsonc
"controller": {
  "id": "Manual_Controller",
  "type": "manual-controller-api",
  "manual-controller-api-settings": {
    "actuator-order": [
      {"id": "Prop_FR_actuator", "initial-value": 0.0},
      {"id": "Prop_RL_actuator", "initial-value": 0.0},
      {"id": "Prop_FL_actuator", "initial-value": 0.0},
      {"id": "Prop_RR_actuator", "initial-value": 0.0}
    ]
  }
}
```

Không thay đổi các file scene và robot đang được `velocity_ned/env.py` sử dụng.

Tiêu chí hoàn thành:

- Config mới hợp lệ theo schema ProjectAirSim.
- Bốn actuator ID khớp chính xác với phần `actuators`.
- Scene velocity-level cũ không bị thay đổi.

### Giai đoạn 2: Viết smoke test cho actuator

Tạo script:

```text
drone_control/action_spaces/motor_residual/check_actuators.py
```

Chức năng của script:

1. Load scene motor-level.
2. Gửi cùng một control signal nhỏ tới cả bốn rotor.
3. Kiểm tra từng actuator độc lập theo thứ tự FL, FR, RL, RR.
4. Xác nhận `set_control_signals()` trả về thành công.
5. Gửi `0.0` tới cả bốn rotor trong khối `finally`.

Nên dùng dictionary theo actuator ID, không dựa vào vị trí trong `actuator-order`:

```python
{
    "Prop_FL_actuator": u_fl,
    "Prop_FR_actuator": u_fr,
    "Prop_RL_actuator": u_rl,
    "Prop_RR_actuator": u_rr,
}
```

Tiêu chí hoàn thành:

- Xác minh đúng mapping giữa action index và rotor vật lý.
- Không có rotor nào tiếp tục quay sau khi script kết thúc.
- Có kiểm tra lỗi khi actuator ID không hợp lệ.

### Giai đoạn 3: Hiệu chuẩn tốc độ hover

Tạo script:

```text
drone_control/action_spaces/motor_residual/calibrate.py
```

Mục tiêu là xác định `hover_control_signal` và `hover_rpm` thực tế. Giá trị ước tính ban đầu từ cấu hình hiện tại:

```text
max_rpm             ~= 6396.667 RPM
hover_control_signal ~= 0.72
hover_rpm            ~= 5400 RPM
```

Thực hiện sweep hoặc binary search control signal, sau đó đo vận tốc thẳng đứng trung bình. Giá trị hover là điểm mà vận tốc thẳng đứng gần bằng không.

Kết quả hiệu chuẩn cần được lưu riêng, ví dụ:

```text
config/motor_control.json
```

Tiêu chí hoàn thành:

- Tìm được operating point khiến drone gần như không tăng hoặc giảm độ cao.
- Ghi lại `max_rpm`, `hover_rpm`, `hover_control_signal` và điều kiện thí nghiệm.
- Không hard-code kết quả đo rải rác trong nhiều file.

### Giai đoạn 4: Tạo environment motor-level

Tạo file:

```text
drone_control/action_spaces/motor_residual/env.py
```

Không sửa `velocity_ned/env.py` hoặc `velocity_ned/env_short_episode.py` ở giai đoạn này.

Không gian hành động:

```text
action = [a_FL, a_FR, a_RL, a_RR] thuộc [-1, 1]^4
```

Ánh xạ action sang tốc độ rotor:

```python
rpm = np.clip(
    hover_rpm + action * rpm_delta,
    min_rpm,
    max_rpm,
)
motor_signal = np.square(rpm / max_rpm)
```

Tần số điều khiển khởi đầu:

```text
CONTROL_HZ = 50
dt = 0.02 giây
```

Observation đề xuất gồm 17 phần tử:

```text
[x, y, z,
 roll, pitch, yaw,
 vx, vy, vz,
 roll_rate, pitch_rate, yaw_rate,
 dist_to_goal,
 previous_motor_FL, previous_motor_FR,
 previous_motor_RL, previous_motor_RR]
```

Nếu `rotor_info` được simulator publish ổn định, có thể thay previous command bằng tốc độ rotor đo được. Nếu không, dùng lệnh motor của bước trước.

Tiêu chí hoàn thành:

- `observation_space.shape == (17,)`.
- `action_space.shape == (4,)` và miền action là `[-1, 1]`.
- `step()` gửi đúng bốn actuator ID.
- Lock-step dựa trên sim time vẫn hoạt động.
- Có timeout khi chờ pose để tránh vòng lặp vô hạn.
- Reset và close luôn đưa motor về `0.0`.

### Giai đoạn 5: Viết unit test cho environment

Tạo thư mục hoặc file test phù hợp, kiểm tra tối thiểu:

- Action `[-1, 1]` được clip đúng.
- `action = zeros(4)` ánh xạ về operating point hover.
- RPM không vượt `min_rpm` và `max_rpm`.
- Chuyển đổi `RPM -> control signal` đúng công thức bình phương.
- Thứ tự FL, FR, RL, RR không bị đảo.
- `prev_action` có kích thước 4.
- Observation có kích thước 17.
- `set_control_signals()` thất bại làm environment báo lỗi rõ ràng.
- Motor được dừng khi exception xảy ra.

Các test này nên dùng mock; không tự động khởi động simulator.

### Giai đoạn 6: Refactor PPO để hỗ trợ action bounds

Loại bỏ phụ thuộc hard-code vào `ScaleAction(1.5)` và clamp `[-1.5, 1.5]` trong `drone_control/algorithms/ppo.py`.

PPO nên nhận cấu hình action bounds hoặc luôn sinh action chuẩn hóa `[-1, 1]`, còn environment chịu trách nhiệm chuyển sang đơn vị vật lý.

Cấu hình motor-level:

```text
STATE_DIM  = 17
ACTION_DIM = 4
```

Hướng triển khai:

- Prototype: Gaussian action rồi clip về `[-1, 1]`.
- Bản chính thức: tanh-squashed Gaussian có hiệu chỉnh `log_prob`.

Tiêu chí hoàn thành:

- PPO không còn giả định action có ba chiều.
- PPO không còn giả định giới hạn action là `1.5`.
- Test forward pass cho cả cấu hình velocity `(13, 3)` và motor `(17, 4)`.
- Workflow velocity-level cũ vẫn có thể khởi tạo khi truyền đúng cấu hình.

### Giai đoạn 7: Thiết kế reward motor-level

Reward đề xuất:

```text
r =
  + w_pos    * exp(-k_pos * ||position_error||^2)
  + w_alive
  - w_vel    * ||linear_velocity||^2
  - w_att    * (roll^2 + pitch^2)
  - w_rate   * ||angular_velocity||^2
  - w_motor  * mean((u - u_hover)^2)
  - w_smooth * mean((u - u_prev)^2)
```

Bổ sung:

- Phạt yaw error hoặc yaw rate.
- Phạt motor saturation tại `0` hoặc `1`.
- Crash penalty lớn hơn đáng kể reward của một bước.
- Kết thúc sớm khi roll hoặc pitch vượt khoảng 60-75 độ.

Tiêu chí hoàn thành:

- Mỗi reward term được log riêng trong `info`.
- Không có term nào phụ thuộc sai đơn vị RPM/control signal.
- Có test reward cho hover, nghiêng, rơi và motor saturation.

### Giai đoạn 8: Tạo curriculum training

Không bắt đầu bằng bài toán takeoff hoàn chỉnh. Triển khai theo thứ tự:

1. Giữ attitude tại `z = -2 m`, khởi tạo gần trạng thái hover.
2. Random roll và pitch nhỏ.
3. Random position và velocity nhỏ.
4. Học thay đổi độ cao.
5. Học takeoff từ gần mặt đất.
6. Thêm disturbance.
7. Thêm domain randomization.

Domain randomization đề xuất:

- Mass: ±5-10%.
- `coeff-of-thrust`.
- Motor time constant.
- Sensor noise.
- Action delay.
- Sai lệch `hover_control_signal`.

Tiêu chí chuyển curriculum:

- Tỷ lệ sống sót đạt ngưỡng đặt trước.
- Position RMSE và attitude RMSE đạt ngưỡng.
- Không dựa riêng vào tổng episode reward.

### Giai đoạn 9: Tạo training entry point riêng

Tạo file:

```text
drone_control/action_spaces/motor_residual/train.py
```

Checkpoint và log phải nằm trong namespace riêng:

```text
log_dir/hover_motor/
checkpoints_motor/
```

Metadata cần lưu cùng checkpoint:

- `state_dim` và `action_dim`.
- Thứ tự rotor.
- `max_rpm`, `hover_rpm`, `rpm_delta`.
- Observation layout.
- Reward weights.
- Control frequency.
- Git commit hoặc mã nhận diện phiên bản source nếu có.

Không overwrite checkpoint velocity-level hiện tại.

### Giai đoạn 10: Tạo workflow đánh giá và điều khiển

Tạo các file riêng:

```text
ControlDrone_motor.py
evaluate_balance_motor.py
```

`velocity_ned/evaluate_balance.py` hiện dùng một velocity action làm disturbance. Cách này không còn đúng với motor environment. Disturbance mới phải sử dụng external force/impulse hoặc cơ chế websocket riêng, không truyền kick velocity vào `env.step()`.

Các metric nên ghi thêm:

- RPM/control signal của từng rotor.
- Motor saturation ratio.
- Position RMSE.
- Roll/pitch/yaw RMSE.
- Angular-rate RMSE.
- Recovery time sau disturbance.
- Năng lượng hoặc tổng control effort ước tính.

### Giai đoạn 11: Chuẩn bị cho PX4 hoặc drone thật

`velocity_ned/control_mavsdk.py` hiện gửi `VelocityNedYaw`, vì vậy không thể dùng trực tiếp checkpoint motor-level.

Trước khi triển khai motor policy lên PX4 cần:

- Xác định actuator-control API tương thích với phiên bản PX4/MAVSDK.
- Xác minh thứ tự motor của PX4 và ProjectAirSim.
- Thêm arm interlock, kill switch và failsafe timeout.
- Giới hạn motor output an toàn.
- Test Software-in-the-Loop trước Hardware-in-the-Loop.
- Không thử trực tiếp trên drone thật khi chưa có test bench và cơ chế dừng khẩn cấp.

## 5. Tương thích checkpoint

Checkpoint velocity-level hiện tại không tương thích trực tiếp vì:

- Actor output đổi từ 3 sang 4.
- State có thể đổi từ 13 sang 17.
- Ý nghĩa action đổi từ velocity setpoint sang rotor command.

Có thể thử chuyển một phần trọng số feature extractor, nhưng không nên xem đó là resume training. Phương án mặc định là train model motor-level mới từ đầu.

## 6. Danh sách file dự kiến

File mới:

```text
sim_config/scene_basic_drone_motor.jsonc
sim_config/robot_quadrotor_motor.jsonc
config/motor_control.json
drone_control/action_spaces/motor_residual/env.py
drone_control/action_spaces/motor_residual/check_actuators.py
drone_control/action_spaces/motor_residual/calibrate.py
drone_control/action_spaces/motor_residual/train.py
ControlDrone_motor.py
evaluate_balance_motor.py
```

File cần refactor có kiểm soát:

```text
drone_control/algorithms/ppo.py
```

File giữ nguyên trong giai đoạn đầu:

```text
drone_control/action_spaces/velocity_ned/env.py
drone_control/action_spaces/velocity_ned/env_short_episode.py
drone_control/action_spaces/velocity_ned/train.py
drone_control/action_spaces/velocity_ned/train_short_episode.py
drone_control/action_spaces/velocity_ned/control_projectairsim.py
drone_control/action_spaces/velocity_ned/control_disturbance.py
drone_control/action_spaces/velocity_ned/evaluate_balance.py
drone_control/action_spaces/velocity_ned/control_mavsdk.py
```

## 7. Checklist nghiệm thu tổng thể

- [ ] Config manual-controller load thành công.
- [ ] Mapping bốn rotor được xác minh.
- [ ] Có cơ chế dừng motor trong mọi đường thoát.
- [ ] Đã hiệu chuẩn hover operating point.
- [ ] Environment motor vượt qua unit test không cần simulator.
- [ ] Action và observation dimensions không còn hard-code sai.
- [ ] Policy action được giới hạn đúng.
- [ ] Reward terms được log và kiểm thử riêng.
- [ ] Policy giữ hover ổn định ở curriculum stage đầu.
- [ ] Disturbance evaluation không dùng velocity action giả.
- [ ] Checkpoint motor có metadata đầy đủ.
- [ ] Checkpoint và kết quả cũ không bị ghi đè.
- [ ] Đã chạy `codegraph sync .` sau thay đổi source.

## 8. Thứ tự ưu tiên ngắn gọn

```text
1. Config manual-controller
2. Smoke test actuator
3. Calibrate hover signal
4. Implement `motor_residual/env.py`
5. Unit test environment
6. Refactor PPO action bounds
7. Implement reward
8. Implement curriculum
9. Add `motor_residual/train.py`
10. Add motor-level control/evaluation
11. Chỉ sau cùng mới xem xét PX4 hoặc drone thật
```
