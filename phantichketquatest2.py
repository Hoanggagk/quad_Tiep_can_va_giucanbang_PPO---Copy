"""
Phân tích kết quả bay từ file trajectory CSV.

Các chỉ số được tính:
  Settling time (z-axis)    – Thời điểm sai số 3D vị trí lần đầu tiên
                               và duy trì < 5 % biên độ leo của trục z.
  Overshoot (z-axis)        – Sai số tĩnh trục z so với target:
                               (z_target − z_final) / z_target × 100 %
  RMSE vị trí theo trục x   – sqrt( mean(x²) ) toàn episode
  RMSE vị trí theo trục y   – sqrt( mean(y²) ) toàn episode
  Sai lệch roll cực đại     – max |roll|
  Sai lệch pitch cực đại    – max |pitch|
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
CSV_PATH   = r"control_results\PPO4R\trajectory_ep001.csv"
TARGET_Z   = 2.0   # m  (goal_pose = [0, 0, −2] NED → altitude = 2 m)
SETTLE_PCT = 0.05  # ngưỡng settling: 5 % biên độ bước nhảy z
OUTPUT_DIR = r"test_results\Co_nhieu\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Đọc dữ liệu ─────────────────────────────────────────────────────────────
df    = pd.read_csv(CSV_PATH)
t     = df["time_s"].values
x     = df["x_m"].values
y     = df["y_m"].values
z     = df["z_m"].values
roll  = df["roll_deg"].values
pitch = df["pitch_deg"].values

# ─── Tính các chỉ số ─────────────────────────────────────────────────────────

# 1. Settling time (z-axis)
#    Sai số 3D = sqrt(x² + y² + (z − z_target)²)
#    Settling khi sai số 3D < 5 % × |z_target − z(0)| và giữ nguyên đến cuối.
step_amp     = abs(TARGET_Z - z[0])          # biên độ bước nhảy z
threshold_3d = SETTLE_PCT * step_amp         # ngưỡng (m)
dist_3d      = np.sqrt(x**2 + y**2 + (z - TARGET_Z)**2)

settling_time = None
for i in range(len(t)):
    if np.all(dist_3d[i:] < threshold_3d):
        settling_time = float(t[i])
        break

# 2. Overshoot (z-axis) – sai số tĩnh tương đối so với target
#    Dương: drone chưa đạt đủ độ cao (undershoot/steady-state error)
z_final  = float(z[-1])
overshoot = (TARGET_Z - z_final) / TARGET_Z * 100.0   # %

# 3. RMSE vị trí x và y  (target x = y = 0)
rmse_x = float(np.sqrt(np.mean(x**2)))
rmse_y = float(np.sqrt(np.mean(y**2)))

# 4. Sai lệch roll và pitch cực đại
max_roll  = float(np.max(np.abs(roll)))
max_pitch = float(np.max(np.abs(pitch)))

# ─── In kết quả ───────────────────────────────────────────────────────────────
W = 54
SEP = "─" * W
print(f"\n{'PHÂN TÍCH KẾT QUẢ BAY':^{W}}")
print(f"  File : {CSV_PATH}")
print(f"  Target : x=0 m, y=0 m, z={TARGET_Z} m")
print(SEP)
print(f"  {'Chỉ số đánh giá':<32} {'Giá trị':>18}")
print(SEP)

st_str = f"{settling_time:.1f} s" if settling_time is not None else "Không hội tụ"
print(f"  {'Settling time (z-axis)':<32} {st_str:>18}")
print(f"  {'Overshoot (z-axis)':<32} {f'{overshoot:.1f} %':>18}")
print(f"  {'RMSE vị trí theo trục x':<32} {f'{rmse_x:.3f} m':>18}")
print(f"  {'RMSE vị trí theo trục y':<32} {f'{rmse_y:.3f} m':>18}")
print(f"  {'Sai lệch roll cực đại':<32} {f'{max_roll:.2f}°':>18}")
print(f"  {'Sai lệch pitch cực đại':<32} {f'{max_pitch:.2f}°':>18}")
print(SEP)
print(f"  Ghi chú: Settling threshold = {SETTLE_PCT*100:.0f}% × biên độ z = {threshold_3d:.4f} m")
print(SEP + "\n")

# ─── Vẽ đồ thị ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"Phân tích quỹ đạo – {os.path.basename(CSV_PATH)}",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)


def _vline(ax):
    """Đường thẳng đứng tại thời điểm hội tụ."""
    if settling_time is not None:
        ax.axvline(settling_time, color="red", linestyle="--",
                   linewidth=1.2, label=f"Settling: {settling_time:.1f} s")


# ── Trục z ──────────────────────────────────────────────────────────────────
ax_z = fig.add_subplot(gs[0, :])
ax_z.plot(t, z, "b-", linewidth=1.5, label="z thực tế")
ax_z.axhline(TARGET_Z, color="k", linestyle="--", linewidth=1,
             label=f"Mục tiêu z = {TARGET_Z} m")
ax_z.axhspan(TARGET_Z - threshold_3d, TARGET_Z + threshold_3d,
             alpha=0.12, color="green",
             label=f"±{SETTLE_PCT*100:.0f}% band ({threshold_3d:.3f} m)")
_vline(ax_z)
if settling_time is not None:
    idx = np.argmin(np.abs(t - settling_time))
    ax_z.plot(settling_time, z[idx], "r^", markersize=9, zorder=5,
              label=f"z tại settling = {z[idx]:.3f} m")
ax_z.set_xlabel("Thời gian (s)")
ax_z.set_ylabel("z (m)")
ax_z.set_title(
    f"Trục z  |  Settling time = {st_str}  |  Overshoot = {overshoot:.1f} %"
)
ax_z.legend(fontsize=8, loc="lower right")
ax_z.grid(True, alpha=0.4)

# ── Trục x ──────────────────────────────────────────────────────────────────
ax_x = fig.add_subplot(gs[1, 0])
ax_x.plot(t, x, color="forestgreen", linewidth=1.5, label="x thực tế")
ax_x.axhline(0, color="k", linestyle="--", linewidth=1)
ax_x.axhline( rmse_x, color="orange", linestyle=":", linewidth=1.2,
              label=f"RMSE = {rmse_x:.3f} m")
ax_x.axhline(-rmse_x, color="orange", linestyle=":", linewidth=1.2)
_vline(ax_x)
ax_x.set_xlabel("Thời gian (s)")
ax_x.set_ylabel("x (m)")
ax_x.set_title(f"Trục x  |  RMSE = {rmse_x:.3f} m")
ax_x.legend(fontsize=8)
ax_x.grid(True, alpha=0.4)

# ── Trục y ──────────────────────────────────────────────────────────────────
ax_y = fig.add_subplot(gs[1, 1])
ax_y.plot(t, y, color="purple", linewidth=1.5, label="y thực tế")
ax_y.axhline(0, color="k", linestyle="--", linewidth=1)
ax_y.axhline( rmse_y, color="orange", linestyle=":", linewidth=1.2,
              label=f"RMSE = {rmse_y:.3f} m")
ax_y.axhline(-rmse_y, color="orange", linestyle=":", linewidth=1.2)
_vline(ax_y)
ax_y.set_xlabel("Thời gian (s)")
ax_y.set_ylabel("y (m)")
ax_y.set_title(f"Trục y  |  RMSE = {rmse_y:.3f} m")
ax_y.legend(fontsize=8)
ax_y.grid(True, alpha=0.4)

# ── Roll ─────────────────────────────────────────────────────────────────────
ax_r = fig.add_subplot(gs[2, 0])
ax_r.plot(t, roll, color="darkcyan", linewidth=1.5, label="Roll")
ax_r.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_r.axhline( max_roll, color="red", linestyle=":", linewidth=1.2,
              label=f"Max = {max_roll:.2f}°")
ax_r.axhline(-max_roll, color="red", linestyle=":", linewidth=1.2)
ax_r.set_xlabel("Thời gian (s)")
ax_r.set_ylabel("Góc (°)")
ax_r.set_title(f"Roll  |  Sai lệch cực đại = {max_roll:.2f}°")
ax_r.legend(fontsize=8)
ax_r.grid(True, alpha=0.4)

# ── Pitch ────────────────────────────────────────────────────────────────────
ax_p = fig.add_subplot(gs[2, 1])
ax_p.plot(t, pitch, color="darkorange", linewidth=1.5, label="Pitch")
ax_p.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_p.axhline( max_pitch, color="red", linestyle=":", linewidth=1.2,
              label=f"Max = {max_pitch:.2f}°")
ax_p.axhline(-max_pitch, color="red", linestyle=":", linewidth=1.2)
ax_p.set_xlabel("Thời gian (s)")
ax_p.set_ylabel("Góc (°)")
ax_p.set_title(f"Pitch  |  Sai lệch cực đại = {max_pitch:.2f}°")
ax_p.legend(fontsize=8)
ax_p.grid(True, alpha=0.4)

# ── Lưu và hiển thị ──────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "phan_tich_ep002.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Đã lưu đồ thị: {out_path}")
