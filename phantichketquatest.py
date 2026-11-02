"""
Phân tích kết quả bay từ file trajectory CSV.
Các chỉ số được tính:
  - Thời gian quá độ (Settling time)  : thời điểm sai số 3D lần đầu tiên
                                         và duy trì < 5% biên độ bước nhảy z
  - Độ quá điều chỉnh (Overshoot)     : (z_target − z_final) / z_target × 100%
  - Sai lệch RMSE theo trục x, y, z   : sqrt(mean(error²)) cho từng trục
  - Sai lệch cực đại Roll, Pitch, Yaw : max |angle|
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
CSV_PATH   = r"control_results\PPO4R\trajectory_ep001.csv"
TARGET_X   = 0.0   # m
TARGET_Y   = 0.0   # m
TARGET_Z   = 2.0   # m  (goal_pose = [0,0,-2] NED → altitude = 2 m)
SETTLE_PCT = 0.05  # ngưỡng settling: 5% biên độ bước nhảy z
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
yaw   = df["yaw_deg"].values

# ─── Tính các chỉ số ─────────────────────────────────────────────────────────

# 1. Thời gian quá độ (Settling time) dựa trên VỊ TRÍ
#    Sai số 3D vị trí = sqrt(x² + y² + (z − z_target)²)
#    Settling khi sai số < 5% × |z_target − z(0)| và giữ nguyên đến cuối
step_amp      = abs(TARGET_Z - z[0])         # biên độ bước nhảy z
threshold_pos = SETTLE_PCT * step_amp        # ngưỡng vị trí (m)
dist_3d_pos   = np.sqrt(x**2 + y**2 + (z - TARGET_Z)**2)

settling_time_pos = None
for i in range(len(t)):
    if np.all(dist_3d_pos[i:] < threshold_pos):
        settling_time_pos = float(t[i])
        break

# 1b. Thời gian quá độ dựa trên GÓC
#     Sai số 3D góc = sqrt(roll² + pitch² + yaw²)
#     Settling khi sai số góc < 5% của sai số góc cực đại
threshold_ang = SETTLE_PCT * np.max(np.sqrt(roll**2 + pitch**2 + yaw**2))
dist_3d_ang   = np.sqrt(roll**2 + pitch**2 + yaw**2)

settling_time_ang = None
for i in range(len(t)):
    if np.all(dist_3d_ang[i:] < threshold_ang):
        settling_time_ang = float(t[i])
        break

# 2. Độ quá điều chỉnh (Overshoot) – tương đối so với target z
#    Overshoot = (z_target − z_final) / z_target × 100 %
z_final   = float(z[-1])
overshoot = (TARGET_Z - z_final) / TARGET_Z * 100.0

# 3. Sai lệch RMSE theo từng trục vị trí
rmse_x = float(np.sqrt(np.mean((x - TARGET_X)**2)))
rmse_y = float(np.sqrt(np.mean((y - TARGET_Y)**2)))
rmse_z = float(np.sqrt(np.mean((z - TARGET_Z)**2)))

# 3b. Thời gian phục hồi sau sai lệch X,Y
#     = thời gian từ lúc sai lệch XY cực đại đến khi trở lại dưới threshold settling
dist_xy = np.sqrt(x**2 + y**2)
max_xy_idx = np.argmax(dist_xy)
threshold_xy = SETTLE_PCT * step_amp  # dùng threshold tương tự settling
recovery_time_xy = None

# Tìm thời gian sau peak mà sai lệch XY quay trở lại dưới threshold
for i in range(max_xy_idx, len(t)):
    if dist_xy[i] < threshold_xy:
        recovery_time_xy = float(t[i] - t[max_xy_idx])
        break

# 4. Sai lệch cực đại Roll, Pitch, Yaw
max_roll  = float(np.max(np.abs(roll)))
max_pitch = float(np.max(np.abs(pitch)))
max_yaw   = float(np.max(np.abs(yaw)))

# 4b. Sai lệch RMSE Roll, Pitch, Yaw (so với target = 0°)
rmse_roll  = float(np.sqrt(np.mean(roll**2)))
rmse_pitch = float(np.sqrt(np.mean(pitch**2)))
rmse_yaw   = float(np.sqrt(np.mean(yaw**2)))

# ─── In kết quả ───────────────────────────────────────────────────────────────
W = 65
SEP = "─" * W
print(f"\n{'PHÂN TÍCH KẾT QUẢ BAY':^{W}}")
print(f"  File : {CSV_PATH}")
print(f"  Mục tiêu: x={TARGET_X} m, y={TARGET_Y} m, z={TARGET_Z} m")
print(f"  Threshold settling: {SETTLE_PCT*100:.0f}% (vị trí: {threshold_pos:.4f} m, góc: {threshold_ang:.2f}°)")
print(SEP + "\n")

# ═════════════════════════════════════════════════════════════════════════════
# KẾT QUẢ 1: PHÂN TÍCH QUỸ ĐẠO VỊ TRÍ (X, Y, Z)
# ═════════════════════════════════════════════════════════════════════════════
print("┌ " + "KẾT QUẢ 1: PHÂN TÍCH QUỸĐẠO VỊ TRÍ (X, Y, Z)".ljust(W-2) + "┐")
print("├" + "─" * (W-2) + "┤")

st_pos_str = f"{settling_time_pos:.1f} s" if settling_time_pos is not None else "Không hội tụ"
print(f"│  {'Thời gian quá độ':<40} {st_pos_str:>20}  │")
print(f"│  {'Độ quá điều chỉnh (z)':<40} {f'{overshoot:.1f} %':>20}  │")
print(f"│  {'Sai lệch RMSE trục x':<40} {f'{rmse_x:.4f} m':>20}  │")
print(f"│  {'Sai lệch RMSE trục y':<40} {f'{rmse_y:.4f} m':>20}  │")
print(f"│  {'Sai lệch RMSE trục z':<40} {f'{rmse_z:.4f} m':>20}  │")

recovery_xy_str = f"{recovery_time_xy:.1f} s" if recovery_time_xy is not None else "Không phục hồi"
print(f"│  {'Thời gian phục hồi sau sai lệch X,Y':<40} {recovery_xy_str:>20}  │")
print("└" + "─" * (W-2) + "┘\n")

# ═════════════════════════════════════════════════════════════════════════════
# KẾT QUẢ 2: PHÂN TÍCH QUỸĐẠO GÓC (ROLL, PITCH, YAW)
# ═════════════════════════════════════════════════════════════════════════════
print("┌ " + "KẾT QUẢ 2: PHÂN TÍCH QUỸĐẠO GÓC (ROLL, PITCH, YAW)".ljust(W-2) + "┐")
print("├" + "─" * (W-2) + "┤")

st_ang_str = f"{settling_time_ang:.1f} s" if settling_time_ang is not None else "Không hội tụ"
print(f"│  {'Thời gian quá độ':<40} {st_ang_str:>20}  │")
print(f"│  {'Độ quá điều chỉnh (z)':<40} {f'{overshoot:.1f} %':>20}  │")
print(f"│  {'Sai lệch RMSE Roll (target=0°)':<40} {f'{rmse_roll:.2f}°':>20}  │")
print(f"│  {'Sai lệch RMSE Pitch (target=0°)':<40} {f'{rmse_pitch:.2f}°':>20}  │")
print(f"│  {'Sai lệch RMSE Yaw (target=0°)':<40} {f'{rmse_yaw:.2f}°':>20}  │")
print("└" + "─" * (W-2) + "┘\n")

# ─── Vẽ đồ thị ──────────────────────────────────────────────────────────────
def _vline_pos(ax):
    """Đường thẳng tại settling vị trí."""
    if settling_time_pos is not None:
        ax.axvline(settling_time_pos, color="red", linestyle="--",
                   linewidth=1.1, alpha=0.7)

def _vline_ang(ax):
    """Đường thẳng tại settling góc."""
    if settling_time_ang is not None:
        ax.axvline(settling_time_ang, color="red", linestyle="--",
                   linewidth=1.1, alpha=0.7)

# ═════════════════════════════════════════════════════════════════════════════
# HÌNH 1: VỊ TRÍ (X, Y, Z)
# ═════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(16, 5))
fig1.suptitle(
    f"KẾT QUẢ 1: Phân tích quỹ đạo vị trí – {os.path.basename(CSV_PATH)}",
    fontsize=13, fontweight="bold"
)
gs1 = gridspec.GridSpec(1, 4, figure=fig1, hspace=0.35, wspace=0.35)

# ── Trục z ──────────────────────────────────────────────────────────────────
ax_z = fig1.add_subplot(gs1[0, 0])
ax_z.plot(t, z, "b-", linewidth=2, label="z thực tế")
ax_z.axhline(TARGET_Z, color="k", linestyle="--", linewidth=1,
             label=f"Mục tiêu z = {TARGET_Z} m")
ax_z.axhspan(TARGET_Z - threshold_pos, TARGET_Z + threshold_pos,
             alpha=0.12, color="green",
             label=f"Vùng settling = {threshold_pos:.4f} m")
_vline_pos(ax_z)
if settling_time_pos is not None:
    idx = np.argmin(np.abs(t - settling_time_pos))
    ax_z.plot(settling_time_pos, z[idx], "r^", markersize=9, zorder=5)
ax_z.set_xlabel("Thời gian (s)", fontsize=10)
ax_z.set_ylabel("z (m)", fontsize=10)
ax_z.set_title(f"Trục Z\nOvershoot = {overshoot:.1f}%  |  RMSE = {rmse_z:.4f} m", 
               fontsize=10, fontweight="bold")
ax_z.legend(fontsize=8, loc="best")
ax_z.grid(True, alpha=0.35)

# ── Trục x ──────────────────────────────────────────────────────────────────
ax_x = fig1.add_subplot(gs1[0, 1])
ax_x.plot(t, x, color="forestgreen", linewidth=2, label="x thực tế")
ax_x.axhline(TARGET_X, color="k", linestyle="--", linewidth=1)
ax_x.axhline( rmse_x, color="orange", linestyle=":", linewidth=1.3,
              label=f"RMSE = {rmse_x:.4f} m")
ax_x.axhline(-rmse_x, color="orange", linestyle=":", linewidth=1.3)
_vline_pos(ax_x)
ax_x.set_xlabel("Thời gian (s)", fontsize=10)
ax_x.set_ylabel("x (m)", fontsize=10)
ax_x.set_title(f"Trục X\nRMSE = {rmse_x:.4f} m", fontsize=10, fontweight="bold")
ax_x.legend(fontsize=8, loc="best")
ax_x.grid(True, alpha=0.35)

# ── Trục y ──────────────────────────────────────────────────────────────────
ax_y = fig1.add_subplot(gs1[0, 2])
ax_y.plot(t, y, color="purple", linewidth=2, label="y thực tế")
ax_y.axhline(TARGET_Y, color="k", linestyle="--", linewidth=1)
ax_y.axhline( rmse_y, color="orange", linestyle=":", linewidth=1.3,
              label=f"RMSE = {rmse_y:.4f} m")
ax_y.axhline(-rmse_y, color="orange", linestyle=":", linewidth=1.3)
_vline_pos(ax_y)
ax_y.set_xlabel("Thời gian (s)", fontsize=10)
ax_y.set_ylabel("y (m)", fontsize=10)
ax_y.set_title(f"Trục Y\nRMSE = {rmse_y:.4f} m", fontsize=10, fontweight="bold")
ax_y.legend(fontsize=8, loc="best")
ax_y.grid(True, alpha=0.35)

# ── Giản đồ quỹ đạo XY ──────────────────────────────────────────────────────
ax_xy = fig1.add_subplot(gs1[0, 3])
ax_xy.plot(x, y, "purple", linewidth=2.5, label="Quỹ đạo XY")
ax_xy.plot(0, 0, "r*", markersize=18, label="Mục tiêu", zorder=5)
ax_xy.plot(x[0], y[0], "go", markersize=9, label="Khởi đầu", zorder=4)
if settling_time_pos is not None:
    idx = np.argmin(np.abs(t - settling_time_pos))
    ax_xy.plot(x[idx], y[idx], "r^", markersize=9, label="Settling", zorder=4)
ax_xy.set_xlabel("x (m)", fontsize=10)
ax_xy.set_ylabel("y (m)", fontsize=10)
ax_xy.set_title("Quỹ đạo hình chiếu XY", fontsize=10, fontweight="bold")
ax_xy.legend(fontsize=8, loc="best")
ax_xy.grid(True, alpha=0.35)
ax_xy.axis("equal")

# ═════════════════════════════════════════════════════════════════════════════
# HÌNH 2: GÓC (ROLL, PITCH, YAW)
# ═════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(15, 5))
fig2.suptitle(
    f"KẾT QUẢ 2: Phân tích quỹ đạo góc – {os.path.basename(CSV_PATH)}",
    fontsize=13, fontweight="bold"
)
gs2 = gridspec.GridSpec(1, 3, figure=fig2, hspace=0.35, wspace=0.35)

# ── Roll ─────────────────────────────────────────────────────────────────────
ax_r = fig2.add_subplot(gs2[0, 0])
ax_r.plot(t, roll, color="darkcyan", linewidth=2, label="Roll")
ax_r.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_r.axhline( rmse_roll, color="orange", linestyle=":", linewidth=1.3,
              label=f"RMSE = {rmse_roll:.2f}°")
ax_r.axhline(-rmse_roll, color="orange", linestyle=":", linewidth=1.3)
_vline_ang(ax_r)
ax_r.set_xlabel("Thời gian (s)", fontsize=10)
ax_r.set_ylabel("Góc (°)", fontsize=10)
ax_r.set_title(f"Roll (target = 0°)\nRMSE = {rmse_roll:.2f}°", fontsize=10, fontweight="bold")
ax_r.legend(fontsize=8, loc="best")
ax_r.grid(True, alpha=0.35)

# ── Pitch ────────────────────────────────────────────────────────────────────
ax_p = fig2.add_subplot(gs2[0, 1])
ax_p.plot(t, pitch, color="darkorange", linewidth=2, label="Pitch")
ax_p.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_p.axhline( rmse_pitch, color="orange", linestyle=":", linewidth=1.3,
              label=f"RMSE = {rmse_pitch:.2f}°")
ax_p.axhline(-rmse_pitch, color="orange", linestyle=":", linewidth=1.3)
_vline_ang(ax_p)
ax_p.set_xlabel("Thời gian (s)", fontsize=10)
ax_p.set_ylabel("Góc (°)", fontsize=10)
ax_p.set_title(f"Pitch (target = 0°)\nRMSE = {rmse_pitch:.2f}°", fontsize=10, fontweight="bold")
ax_p.legend(fontsize=8, loc="best")
ax_p.grid(True, alpha=0.35)

# ── Yaw ──────────────────────────────────────────────────────────────────────
ax_w = fig2.add_subplot(gs2[0, 2])
ax_w.plot(t, yaw, color="darkred", linewidth=2, label="Yaw")
ax_w.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_w.axhline( rmse_yaw, color="orange", linestyle=":", linewidth=1.3,
              label=f"RMSE = {rmse_yaw:.2f}°")
ax_w.axhline(-rmse_yaw, color="orange", linestyle=":", linewidth=1.3)
_vline_ang(ax_w)
ax_w.set_xlabel("Thời gian (s)", fontsize=10)
ax_w.set_ylabel("Góc (°)", fontsize=10)
ax_w.set_title(f"Yaw (target = 0°)\nRMSE = {rmse_yaw:.2f}°", fontsize=10, fontweight="bold")
ax_w.legend(fontsize=8, loc="best")
ax_w.grid(True, alpha=0.35)

# ── Lưu và hiển thị ──────────────────────────────────────────────────────────
out_path1 = os.path.join(OUTPUT_DIR, "KQ1_quytrinh_vitri_ep002.png")
out_path2 = os.path.join(OUTPUT_DIR, "KQ2_quytrinh_goc_ep002.png")

fig1.savefig(out_path1, dpi=150, bbox_inches="tight")
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")

plt.show()
print(f"\n✓ Đã lưu KẾT QUẢ 1: {out_path1}")
print(f"✓ Đã lưu KẾT QUẢ 2: {out_path2}\n")

