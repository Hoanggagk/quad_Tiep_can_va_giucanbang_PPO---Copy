# Báo cáo phân tích huấn luyện PPO 4 động cơ

- Nguồn: `log_dir/hover_motor/motor_20261115_194017_788604/training.csv`
- Số episode: **8,603**
- Timestep cuối: **206,810**
- Cửa sổ so sánh: **500 episode** đầu và cuối

## Kết luận chính

1. **Không có tín hiệu thành công:** toàn bộ episode có `survived=0` và `success=0`, nên curriculum không có điều kiện chuyển stage.
2. **Episode kết thúc quá sớm:** median chỉ 22.0/500 bước. Policy gần như không thu được trajectory hover dài.
3. **Mất ổn định attitude là nguyên nhân có khả năng cao nhất:** attitude RMSE cuối 39.6°, vượt gate 12°. CSV không có `termination_reason`, nên chưa thể khẳng định crash do `attitude_limit` hay một điều kiện kết thúc khác.
4. **Position gate cũng không đạt:** RMSE cuối 0.500 m > 0.30 m.
5. **Reward tổng chủ yếu đo độ dài episode:** tương quan reward–steps là 0.922. Vì vậy reward không tăng rõ không đồng nghĩa PPO loss không học; drone đang không sống đủ lâu để tích lũy reward.
6. **Curriculum bị khóa:** 8,603 episode đều ở `attitude_hover`.
7. **Một phần PPO update quá mạnh:** 14.6% snapshot vượt target KL 0.02. Đây là yếu tố phụ; nó không giải thích riêng việc survival luôn bằng 0.

## So sánh đầu và cuối

| Chỉ số | Đầu | Cuối |
|---|---:|---:|
| Reward/episode trung bình | 12.689 | 14.211 |
| Reward/bước trung bình | 0.6040 | 0.6603 |
| Số bước/episode trung bình | 21.1 | 21.5 |
| Survival rate | 0.0% | 0.0% |
| Success rate | 0.0% | 0.0% |
| Position RMSE (m) | 0.508 | 0.500 |
| Attitude RMSE (độ) | 43.1 | 39.6 |
| Action std | 0.490 | 0.100 |

Tương quan Pearson giữa reward và số bước: **0.922**.

## Curriculum

| Stage | Episode | Tỷ lệ |
|---|---:|---:|
| `attitude_hover` | 8,603 | 100.0% |

## Kết quả theo action standard deviation

| Action std | Episode | Reward TB | Steps TB | Attitude RMSE (độ) | Survival |
|---:|---:|---:|---:|---:|---:|
| 0.4–0.5 | 2,100 | 14.254 | 23.4 | 42.7 | 0.0% |
| 0.3–0.4 | 2,000 | 16.224 | 26.2 | 41.4 | 0.0% |
| 0.2–0.3 | 2,000 | 16.069 | 25.0 | 40.6 | 0.0% |
| 0.1–0.2 | 2,503 | 14.629 | 22.1 | 39.9 | 0.0% |

## PPO diagnostics

Phân tích **103** snapshot PPO khác nhau, loại các giá trị lặp giữa các episode.

| Chỉ số | Median | P95 | Max/cuối |
|---|---:|---:|---:|
| Approx KL | 0.0033 | 0.0312 | 0.0854 |
| Clip fraction | 0.0248 | 0.3301 | 0.5082 |
| Explained variance | 0.9168 | 0.9839 | 0.9846 |

Tỷ lệ update có `approx_kl > 0.02`: **14.6%**.

## Nguyên nhân và hành động đề xuất

1. **Ưu tiên xác nhận termination:** thêm `termination_reason` vào `training.csv`; dữ liệu hiện tại chỉ cho phép suy luận attitude instability từ episode ngắn và attitude RMSE cao.
2. **Không tiếp tục tăng exploration:** action std ban đầu 0.50 trên bốn motor độc lập tạo differential thrust lớn. Hãy thử lại từ checkpoint sạch với std thấp hơn sau khi xác nhận neutral action giữ hover ổn định.
3. **Kiểm tra motor mapping trước PPO:** chạy bài kiểm tra bốn motor cùng action 0 và perturbation đối xứng nhỏ; xác nhận thứ tự FL/FR/RL/RR và chiều moment roll/pitch đúng.
4. **Đánh giá bằng survival và RMSE, không chỉ reward:** reward episode bị chi phối mạnh bởi số bước sống. Curriculum chỉ nên tiến khi survival, success và hai RMSE đạt gate.
5. **Không resume checkpoint hiện tại để kiểm tra cấu hình exploration mới:** policy đã trải qua hàng nghìn episode crash ở stage đầu; dùng run mới giúp tách lỗi policy cũ khỏi lỗi cấu hình/mapping.

## Giới hạn của báo cáo

CSV không lưu `termination_reason`, reward components hoặc action theo từng rotor. Vì vậy báo cáo xác định được triệu chứng và nguyên nhân có khả năng cao, nhưng chưa chứng minh được rotor/moment nào gây mất ổn định.
