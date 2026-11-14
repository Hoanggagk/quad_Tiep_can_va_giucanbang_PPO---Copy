# Residual motor action space

Policy sinh bốn residual action chuẩn hóa trong `[-1, 1]^4`. Environment ánh
xạ chúng thành RPM quanh operating point hover đã hiệu chuẩn trong
`config/motor_control.json`.

Xem bảng cấu hình đầy đủ về state/action, siêu tham số PPO, reward, curriculum
và checkpoint tại [`docs/PPO_CONFIG_TABLE.md`](../../../docs/PPO_CONFIG_TABLE.md#10-cấu-hình-ppo-cho-không-gian-hành-động-4-động-cơ).

- `env.py`: environment và action-to-motor mapping.
- `curriculum.py`: curriculum và domain randomization.
- `train.py`: entry point huấn luyện, checkpoint kèm metadata.
- `monitoring.py`: tiến độ trực tiếp, rolling metrics, throughput và ETA.
- `calibrate.py`: hiệu chuẩn hover control signal; mặc định là dry-run.
- `check_actuators.py`: smoke test actuator; chỉ kết nối khi có `--execute`.

Ví dụ an toàn, không kết nối simulator:

```powershell
python -m drone_control.action_spaces.motor_residual.calibrate --help
python -m drone_control.action_spaces.motor_residual.check_actuators --help
```

Khi huấn luyện, monitor mặc định in tóm tắt mỗi 10 episode và cập nhật hai
artifact trong `log_dir/hover_motor/<run_id>/`:

- `training.csv`: metric từng episode và các cột tiến độ/rolling metrics.
- `training_status.json`: snapshot mới nhất để theo dõi từ chương trình khác.

```powershell
python -m drone_control.action_spaces.motor_residual.train --monitor-every 5 --monitor-window 20
```

Dùng `--quiet-monitor` nếu chỉ muốn ghi artifact mà không in console.
