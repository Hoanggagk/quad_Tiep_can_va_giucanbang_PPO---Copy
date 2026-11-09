# Bảng Tham Số Cấu Hình Thuật Toán PPO

## 1. Tham Số Không Gian (State & Action Space)

| Tham Số        | Ký Hiệu | Giá Trị     | Ý Nghĩa                                                                                                             |
| -------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| `STATE_DIM`    | $N_s$   | 13          | Chiều không gian trạng thái: [x, y, z, roll, pitch, yaw, vx, vy, vz, roll_rate, pitch_rate, yaw_rate, dist_to_goal] |
| `ACTION_DIM`   | $N_a$   | 3           | Chiều không gian hành động: [v_north, v_east, v_down]                                                               |
| `Action Range` | $A$     | [-1.5, 1.5] | c động                                                                                           |

---

## 2. Tham Số Độ Lệch Chuẩn Hành Động (Action Standard Deviation)

| Tham Số                 | Ký Hiệu        | Giá Trị     | Ý Nghĩa                                                             |
| ----------------------- | -------------- | ----------- | ------------------------------------------------------------------- |
| `ACTION_STD_INIT`       | $\sigma_0$     | 0.5         | Độ lệch chuẩn ban đầu (~33% của range ±1.5)                         |
| `MIN_ACTION_STD`        | $\sigma_{min}$ | 0.01        | Độ lệch chuẩn tối thiểu (~1.7% của range, ≈±0.15 m/s khi khai thác) |
| `ACTION_STD_DECAY_RATE` | $\beta$        | 0.01        | Tốc độ giảm std mỗi lần giảm                                        |
| `ACTION_STD_DECAY_FREQ` | $F_\sigma$     | 50 episodes | Tần suất giảm action_std (cứ 50 episode)                            |

**Giải thích**: Độ lệch chuẩn cao ban đầu giúp khám phá (exploration), sau đó giảm dần để khai thác (exploitation).

---

## 3. Tham Số Vòng Lặp Huấn Luyện (Training Loop)

| Tham Số                  | Ký Hiệu   | Giá Trị         | Ý Nghĩa                                              |
| ------------------------ | --------- | --------------- | ---------------------------------------------------- |
| `MAX_TRAINING_TIMESTEPS` | $T_{max}$ | 5×10⁵ (500,000) | Tổng số timestep tối đa                              |
| `STEPS_PER_EPISODE`      | $S_e$     | 50              | Số bước tối đa mỗi episode (50 × 0.1s = 5s thực)     |
| `UPDATE_TIMESTEP`        | $T_{up}$  | 2,000           | Cập nhật PPO sau mỗi N timestep                      |
| `K_EPOCHS`               | $K$       | 30              | Số epoch per update (lặp trên dữ liệu trong 1 batch) |

**Giải thích**:

- Mỗi episode chịu tối đa 50 bước
- Sau mỗi 2,000 timestep (≈40 episodes), cập nhật network PPO
- Mỗi lần update, lặp 30 lần trên dữ liệu batch

---

## 4. Tham Số Thuật Toán PPO (PPO Algorithm)

| Tham Số     | Ký Hiệu    | Giá Trị         | Ý Nghĩa                                                |
| ----------- | ---------- | --------------- | ------------------------------------------------------ |
| `EPS_CLIP`  | $\epsilon$ | 0.2             | Clip hệ số: π_new / π_old ∈ [1-ε, 1+ε] = [0.8, 1.2]    |
| `GAMMA`     | $\gamma$   | 0.99            | Hệ số chiết khấu (discount factor) - giá trị tương lai |
| `LR_ACTOR`  | $\alpha_a$ | 3×10⁻⁴ (0.0003) | Learning rate cho Actor network                        |
| `LR_CRITIC` | $\alpha_c$ | 1×10⁻³ (0.001)  | Learning rate cho Critic network                       |

**Giải thích**:

- `EPS_CLIP = 0.2`: Hạn chế mức độ thay đổi policy trong mỗi update (tránh thay đổi quá tột)
- `GAMMA = 0.99`: Cân bằng giữa reward hiện tại và tương lai
- `LR_CRITIC > LR_ACTOR`: Critic học nhanh hơn để ước lượng value chính xác

---

## 5. Tham Số Kiểm Soát Log (Logging & Checkpointing)

| Tham Số           | Ký Hiệu | Giá Trị               | Ý Nghĩa                             |
| ----------------- | ------- | --------------------- | ----------------------------------- |
| `PRINT_FREQ`      | $F_p$   | 1,000 timesteps       | In log console mỗi N timestep       |
| `LOG_FREQ`        | $F_l$   | 250 timesteps         | Ghi CSV log mỗi N timestep          |
| `SAVE_MODEL_FREQ` | $F_s$   | 50,000 timesteps      | Lưu checkpoint mỗi N timestep       |
| `LOG_DIR`         | —       | "log_dir/"            | Thư mục lưu trữ logs và checkpoints |
| `RUN_NAME`        | —       | "hover_projectairsim" | Tên run (tạo subfolder)             |

---

## 6. Tham Số Môi trường (Environment)

| Tham Số                 | Ký Hiệu      | Giá Trị     | Ý Nghĩa                                   |
| ----------------------- | ------------ | ----------- | ----------------------------------------- |
| `dt`                    | $\Delta t$   | 0.1 seconds | Thời gian giữa các lần gửi lệnh hành động |
| `MAX_STEPS_PER_EPISODE` | $S_{max}$    | 50 bước     | = 50 × 0.1s = 5s thực                     |
| `Goal Pose`             | $\mathbf{g}$ | [0, 0, -2]  | Mục tiêu bay: NED (North, East, Down)     |

---

## 7. Tham Số Reward Function

| Tham Số             | Ký Hiệu    | Giá Trị | Ý Nghĩa                                                  |
| ------------------- | ---------- | ------- | -------------------------------------------------------- |
| `r_pos` weight      | $w_p$      | 1.0     | Thành phần vị trí (1.0 khi ở tâm, giảm đến 0 nếu quá xa) |
| `r_approach` weight | $w_a$      | 2.0     | Thưởng khi tiếp cận mục tiêu                             |
| `r_att` weight      | $w_{att}$  | -0.1    | Phạt nếu drone nghiêng quá nhiều                         |
| `survival bonus`    | $r_s$      | 0.01    | Thưởng sống sót mỗi bước                                 |
| `hover bonus`       | $r_h$      | 0.5     | Thưởng khi hover ở vị trí (<0.2m, <0.2m/s)               |
| `success bonus`     | $r_{succ}$ | 5.0     | Thưởng khi hover thành công >20 bước                     |
| `crash penalty`     | $r_c$      | -2.0    | Phạt khi va chạm hoặc quá xa                             |
| `ground penalty`    | $r_g$      | -2.0    | Phạt khi chạm mặt đất                                    |

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
