# NED velocity action space

Policy sinh ba thành phần vận tốc
`[v_north, v_east, v_down]` trong miền `[-1.5, 1.5]^3` (m/s).

- `env.py`: environment chính.
- `env_short_episode.py`: cùng action space, horizon 100 bước.
- `train.py`, `train_short_episode.py`: entry point huấn luyện.
- `control_projectairsim.py`, `control_mavsdk.py`: triển khai policy.
- `control_disturbance.py`, `evaluate_balance.py`: đánh giá nhiễu và phục hồi.

Chạy entry point từ thư mục gốc bằng `python -m`, ví dụ:

```powershell
python -m drone_control.action_spaces.velocity_ned.train --help
```
