# Residual motor action space

Policy sinh bốn residual action chuẩn hóa trong `[-1, 1]^4`. Environment ánh
xạ chúng thành RPM quanh operating point hover đã hiệu chuẩn trong
`config/motor_control.json`.

- `env.py`: environment và action-to-motor mapping.
- `curriculum.py`: curriculum và domain randomization.
- `train.py`: entry point huấn luyện, checkpoint kèm metadata.
- `calibrate.py`: hiệu chuẩn hover control signal; mặc định là dry-run.
- `check_actuators.py`: smoke test actuator; chỉ kết nối khi có `--execute`.

Ví dụ an toàn, không kết nối simulator:

```powershell
python -m drone_control.action_spaces.motor_residual.calibrate --help
python -m drone_control.action_spaces.motor_residual.check_actuators --help
```
