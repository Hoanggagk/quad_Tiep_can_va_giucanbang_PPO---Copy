"""
KẾT QUẢ 1: PHÂN TÍCH QUỸĐẠO VỊ TRÍ (X, Y, Z)

Các chỉ số được tính:
  - Thời gian quá độ (Settling time)  : thời điểm sai số 3D vị trí lần đầu tiên
                                         và duy trì < 5% biên độ bước nhảy z
  - Độ quá điều chỉnh (Overshoot)     : (z_target − z_final) / z_target × 100%
  - Sai lệch RMSE theo trục x, y, z   : sqrt(mean(error²)) cho từng trục
  - Thời gian phục hồi sau sai lệch X,Y : từ peak đến khi quay về settling band
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
CSV_PATH   = r"control_results\Co_Nhieu\PPO_4R\trajectory_ep001.csv"
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

# 2. Độ quá điều chỉnh (Overshoot) 
z_peak = np.max(z)

overshoot_z = max(
    0.0,
    (z_peak - TARGET_Z) /
    abs(TARGET_Z - z[0]) * 100
)

# Sai số xác lập
steady_error_x = float(x[-1] - TARGET_X)
steady_error_y = float(y[-1] - TARGET_Y)
steady_error_z = float(z[-1] - TARGET_Z)


# 3. Sai lệch RMSE theo từng trục vị trí
rmse_x = float(np.sqrt(np.mean((x - TARGET_X)**2)))
rmse_y = float(np.sqrt(np.mean((y - TARGET_Y)**2)))
rmse_z = float(np.sqrt(np.mean((z - TARGET_Z)**2)))

# 3b. Thời gian phục hồi sau sai lệch X,Y
# ==========================================================
# Thời gian phục hồi trục X
# ==========================================================

threshold_x = SETTLE_PCT * abs(TARGET_Z - z[0])   # hoặc tự đặt ngưỡng cho X

abs_x = np.abs(x - TARGET_X)

peak_x_idx = np.argmax(abs_x)

recovery_time_x = None

for i in range(peak_x_idx, len(t)):
    if np.all(abs_x[i:] < threshold_x):
        recovery_time_x = float(t[i] - t[peak_x_idx])
        break


# ==========================================================
# Thời gian phục hồi trục Y
# ==========================================================

threshold_y = threshold_x

abs_y = np.abs(y - TARGET_Y)

peak_y_idx = np.argmax(abs_y)

recovery_time_y = None

for i in range(peak_y_idx, len(t)):
    if np.all(abs_y[i:] < threshold_y):
        recovery_time_y = float(t[i] - t[peak_y_idx])
        break

# ─── In kết quả ───────────────────────────────────────────────────────────────
W = 65
SEP = "─" * W
print(f"\n{'KẾT QUẢ 1: PHÂN TÍCH QUỸĐẠO VỊ TRÍ (X, Y, Z)':^{W}}")
print(f"  File : {CSV_PATH}")
print(f"  Mục tiêu: x={TARGET_X} m, y={TARGET_Y} m, z={TARGET_Z} m")
print(f"  Threshold settling: {SETTLE_PCT*100:.0f}% (vị trí: {threshold_pos:.4f} m)")
print(SEP)
print(f"  {'Chỉ tiêu':<40} {'Giá trị':>20}")
print(SEP)

st_pos_str = f"{settling_time_pos:.1f} s" if settling_time_pos is not None else "Không hội tụ"
print(f"  {'1. Thời gian quá độ':<40} {st_pos_str:>20}")
print(f"  {'2. Độ quá điều chỉnh (Z)':<40} {f'{overshoot_z:.2f} %':>20}")

print(f"  {'3a. Sai số xác lập trục X':<40} {f'{steady_error_x:.4f} m':>20}")
print(f"  {'3b. Sai số xác lập trục Y':<40} {f'{steady_error_y:.4f} m':>20}")
print(f"  {'3c. Sai số xác lập trục Z':<40} {f'{steady_error_z:.4f} m':>20}")
print(f"  {'3a. Sai lệch RMSE trục x':<40} {f'{rmse_x:.4f} m':>20}")
print(f"  {'3b. Sai lệch RMSE trục y':<40} {f'{rmse_y:.4f} m':>20}")
print(f"  {'3c. Sai lệch RMSE trục z':<40} {f'{rmse_z:.4f} m':>20}")

recovery_x_str = (
    f"{recovery_time_x:.2f} s"
    if recovery_time_x is not None
    else "Không phục hồi"
)

recovery_y_str = (
    f"{recovery_time_y:.2f} s"
    if recovery_time_y is not None
    else "Không phục hồi"
)

print(f"  {'4. Thời gian phục hồi trục X':<40} {recovery_x_str:>20}")
print(f"  {'5. Thời gian phục hồi trục Y':<40} {recovery_y_str:>20}")
print(SEP + "\n")

# ─── Vẽ đồ thị ──────────────────────────────────────────────────────────────
def _vline_pos(ax):
    """Đường thẳng tại settling vị trí."""
    if settling_time_pos is not None:
        ax.axvline(settling_time_pos, color="red", linestyle="--",
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
peak_idx = np.argmax(z)

ax_z.plot(
    t[peak_idx],
    z[peak_idx],
    "ro",
    markersize=7,
    label=f"Peak = {z_peak:.3f} m"
)
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
ax_z.set_title(
    f"Trục Z\nOvershoot = {overshoot_z:.2f}% | RMSE = {rmse_z:.4f} m",
    fontsize=10,
    fontweight="bold"
)

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

# ── Lưu và hiển thị ──────────────────────────────────────────────────────────
out_path1 = os.path.join(OUTPUT_DIR, "KQ1_quytrinh_vitri.png")
fig1.savefig(out_path1, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Đã lưu: {out_path1}\n")
