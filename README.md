# Quadrotor control with PPO

Dự án tách mã nguồn theo **action space** để các phương pháp có thể
phát triển, kiểm thử và so sánh độc lập.

Hướng dẫn đầy đủ: [`docs/HUONG_DAN_SU_DUNG.md`](docs/HUONG_DAN_SU_DUNG.md).

## Các phương pháp

| Phương pháp | Action | Bounds | Mã nguồn |
|---|---|---:|---|
| NED velocity | `[v_north, v_east, v_down]` | `[-1.5, 1.5]^3` m/s | `drone_control/action_spaces/velocity_ned/` |
| Residual motor | residual RPM của `[FL, FR, RL, RR]` | `[-1, 1]^4` | `drone_control/action_spaces/motor_residual/` |

Biến thể `short_episode` thuộc NED velocity vì nó chỉ thay đổi horizon,
không thay đổi ý nghĩa action.

## Cấu trúc

```text
drone_control/
├── algorithms/                  # PPO dùng chung
└── action_spaces/
    ├── velocity_ned/            # velocity control 3D
    └── motor_residual/          # direct motor control 4D
tests/                         # cấu trúc test phản chiếu source
tools/
├── visualization/               # vẽ kết quả offline
└── diagnostics/                 # chẩn đoán simulator/MAVSDK
analysis/legacy/               # script phân tích cũ
docs/                          # tài liệu kỹ thuật
config/                        # calibration/config thuật toán
sim_config/                    # cấu hình ProjectAirSim
```

Các thư mục checkpoint, log và kết quả lịch sử được giữ nguyên để
không làm hỏng đường dẫn của thí nghiệm đã có.

## Chạy

Luôn chạy module từ thư mục gốc:

```powershell
python -m drone_control.action_spaces.velocity_ned.train --help
python -m drone_control.action_spaces.motor_residual.train --help
python -m unittest discover -v
```

Lệnh training và control sẽ kết nối ProjectAirSim; chỉ chạy khi simulator
đã được chuẩn bị. Test tĩnh và `--help` không khởi động training.
