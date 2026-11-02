"""
KẾT QUẢ 2: PHÂN TÍCH QUỸĐẠO GÓC (ROLL, PITCH, YAW)

Các chỉ số được tính (RIÊNG CHO TỪNG TRỤC):
  - Thời gian nằm trong vùng cân bằng:
        tổng thời gian mà |góc| của TRỤC ĐÓ nằm trong ngưỡng cân bằng
        cố định (mặc định 0.5°), tính trên toàn bộ quỹ đạo
        (không nhất thiết liên tục)
  - Sai lệch RMSE Roll, Pitch, Yaw    : sqrt(mean(angle²)) với target = 0°
  - Thời gian phục hồi sau va chạm    : từ thời điểm va chạm (t = 4.4s)
                                         đến khi |góc| quay về và duy trì
                                         trong vùng cân bằng
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
CSV_PATH        = r"control_results\Co_Nhieu\PPO_4R\trajectory_ep001.csv"
EQUIL_THRESHOLD = 0.5  # ngưỡng cân bằng cố định (độ) - áp dụng chung cho cả 3 trục
COLLISION_TIME  = 4.4  # s - thời điểm va chạm (mốc bắt đầu tính phục hồi)
OUTPUT_DIR      = r"test_results\Co_nhieu\plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Đọc dữ liệu ─────────────────────────────────────────────────────────────
df    = pd.read_csv(CSV_PATH)
t     = df["time_s"].values
roll  = df["roll_deg"].values
pitch = df["pitch_deg"].values
yaw   = df["yaw_deg"].values

idx_collision = int(np.searchsorted(t, COLLISION_TIME))
idx_collision = min(idx_collision, len(t) - 1)
t_collision_actual = float(t[idx_collision])  # mốc thời gian thực tế gần nhất trong dữ liệu

# ─── Hàm tính thời gian nằm trong vùng cân bằng cho MỘT trục ──────────────────
def compute_time_in_band(t, signal, threshold=EQUIL_THRESHOLD):
    """
    Tính tổng thời gian |signal| nằm trong ngưỡng cân bằng cố định.

    - mask         : mảng bool, True tại các điểm |signal| < threshold
    - time_in_band : tổng thời gian (s) tích lũy trong vùng cân bằng
                      (cộng dồn dt tại các điểm mask = True, không cần liên tục)
    - percent      : tỷ lệ % thời gian trong vùng cân bằng so với tổng thời gian

    Trả về dict: mask, time_in_band, percent, total_time
    """
    abs_sig    = np.abs(signal)
    mask       = abs_sig < threshold
    dt         = np.diff(t, prepend=t[0])
    dt[0]      = t[1] - t[0] if len(t) > 1 else 0.0

    time_in_band = float(np.sum(dt[mask]))
    total_time   = float(t[-1] - t[0])
    percent      = (time_in_band / total_time * 100.0) if total_time > 0 else 0.0

    return {
        "mask": mask,
        "time_in_band": time_in_band,
        "total_time": total_time,
        "percent": percent,
    }

# ─── Hàm tính thời gian phục hồi sau va chạm cho MỘT trục ─────────────────────
def compute_recovery(t, signal, idx_start, collision_time, threshold=EQUIL_THRESHOLD):
    """
    Tính thời gian phục hồi về vùng cân bằng, tính từ thời điểm va chạm.

    - idx_start      : chỉ số dữ liệu tại/ sau thời điểm va chạm, bắt đầu tìm từ đây
    - t_settled       : thời điểm |signal| quay về và duy trì < threshold đến hết
    - recovery_time   : t_settled - collision_time (None nếu không phục hồi)

    Trả về dict: t_settled, recovery_time
    """
    abs_sig = np.abs(signal)

    t_settled     = None
    recovery_time = None
    for i in range(idx_start, len(t)):
        if np.all(abs_sig[i:] < threshold):
            t_settled     = float(t[i])
            recovery_time = t_settled - collision_time
            break

    return {"t_settled": t_settled, "recovery_time": recovery_time}

# ─── Tính các chỉ số cho từng trục ─────────────────────────────────────────────
band_roll  = compute_time_in_band(t, roll)
band_pitch = compute_time_in_band(t, pitch)
band_yaw   = compute_time_in_band(t, yaw)

recov_roll  = compute_recovery(t, roll,  idx_collision, COLLISION_TIME)
recov_pitch = compute_recovery(t, pitch, idx_collision, COLLISION_TIME)
recov_yaw   = compute_recovery(t, yaw,   idx_collision, COLLISION_TIME)

# RMSE (so với target = 0°)
rmse_roll  = float(np.sqrt(np.mean(roll**2)))
rmse_pitch = float(np.sqrt(np.mean(pitch**2)))
rmse_yaw   = float(np.sqrt(np.mean(yaw**2)))

# ─── Giá trị đỉnh (peak) của từng trục ─────────────────────────────────────────
def compute_peak(t, signal):
    """Trả về (giá trị đỉnh có dấu, |đỉnh|, thời điểm xảy ra đỉnh)."""
    idx_peak   = int(np.argmax(np.abs(signal)))
    peak_signed = float(signal[idx_peak])
    peak_abs    = float(np.abs(peak_signed))
    t_peak      = float(t[idx_peak])
    return peak_signed, peak_abs, t_peak

peak_roll_signed,  peak_roll_abs,  t_peak_roll  = compute_peak(t, roll)
peak_pitch_signed, peak_pitch_abs, t_peak_pitch = compute_peak(t, pitch)
peak_yaw_signed,   peak_yaw_abs,   t_peak_yaw   = compute_peak(t, yaw)

# ─── In kết quả ───────────────────────────────────────────────────────────────
W = 65
SEP = "─" * W

def _band_str(band):
    return f"{band['time_in_band']:.1f}s ({band['percent']:.1f}%)"

def _recov_str(recov):
    return f"{recov['recovery_time']:.2f} s" if recov["recovery_time"] is not None else "Không phục hồi"

print(f"\n{'KẾT QUẢ 2: PHÂN TÍCH QUỸĐẠO GÓC (ROLL, PITCH, YAW)':^{W}}")
print(f"  File : {CSV_PATH}")
print(f"  Mục tiêu: Roll = 0°, Pitch = 0°, Yaw = 0°")
print(f"  Ngưỡng cân bằng: {EQUIL_THRESHOLD:.2f}° (cố định, áp dụng cho cả 3 trục)")
print(f"  Thời điểm va chạm: {COLLISION_TIME:.1f} s (mốc dữ liệu gần nhất: {t_collision_actual:.2f} s)")
print(SEP)
print(f"  {'Chỉ tiêu':<40} {'Giá trị':>20}")
print(SEP)

print(f"  {'1. Thời gian Roll trong vùng cân bằng':<40} {_band_str(band_roll):>20}")
print(f"  {'2. Thời gian Pitch trong vùng cân bằng':<40} {_band_str(band_pitch):>20}")
print(f"  {'3. Thời gian Yaw trong vùng cân bằng':<40} {_band_str(band_yaw):>20}")
print(SEP)
print(f"  {'4a. Sai lệch RMSE Roll':<40} {f'{rmse_roll:.2f}°':>20}")
print(f"  {'4b. Sai lệch RMSE Pitch':<40} {f'{rmse_pitch:.2f}°':>20}")
print(f"  {'4c. Sai lệch RMSE Yaw':<40} {f'{rmse_yaw:.2f}°':>20}")
print(SEP)
print(f"  {'5a. Giá trị đỉnh Roll':<40} {f'{peak_roll_signed:.2f}° @ t={t_peak_roll:.1f}s':>20}")
print(f"  {'5b. Giá trị đỉnh Pitch':<40} {f'{peak_pitch_signed:.2f}° @ t={t_peak_pitch:.1f}s':>20}")
print(f"  {'5c. Giá trị đỉnh Yaw':<40} {f'{peak_yaw_signed:.2f}° @ t={t_peak_yaw:.1f}s':>20}")
print(SEP)
print(f"  {'6a. Thời gian phục hồi Roll (từ va chạm)':<40} {_recov_str(recov_roll):>20}")
print(f"  {'6b. Thời gian phục hồi Pitch (từ va chạm)':<40} {_recov_str(recov_pitch):>20}")
print(f"  {'6c. Thời gian phục hồi Yaw (từ va chạm)':<40} {_recov_str(recov_yaw):>20}")
print(SEP + "\n")

# ─── Vẽ đồ thị ──────────────────────────────────────────────────────────────
def _shade_band(ax, t, band, color="green"):
    """Tô nền các khoảng thời gian mà |góc| nằm trong vùng cân bằng."""
    mask = band["mask"]
    if not np.any(mask):
        return
    # Tìm các đoạn liên tục có mask = True để axvspan cho gọn
    idx = np.where(mask)[0]
    start = idx[0]
    prev  = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            ax.axvspan(t[start], t[prev], color=color, alpha=0.12, linewidth=0)
            start = i
        prev = i
    ax.axvspan(t[start], t[prev], color=color, alpha=0.12, linewidth=0)

def _mark_peak(ax, t_peak, peak_signed):
    """Đánh dấu điểm đỉnh (giá trị cực trị) trên đồ thị."""
    ax.plot(t_peak, peak_signed, marker="o", color="purple", markersize=6, zorder=5)
    ax.axvline(t_peak, color="purple", linestyle="--", linewidth=1.0, alpha=0.5)

def _vline_collision(ax):
    """Đường thẳng tại thời điểm va chạm."""
    ax.axvline(COLLISION_TIME, color="magenta", linestyle="-.",
               linewidth=1.2, alpha=0.8)

def _vline_settled(ax, t_settled):
    """Đường thẳng tại thời điểm phục hồi xong (sau va chạm)."""
    if t_settled is not None:
        ax.axvline(t_settled, color="brown", linestyle="--",
                   linewidth=1.1, alpha=0.7)

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
ax_r.axhline( EQUIL_THRESHOLD, color="green", linestyle=":", linewidth=1.3,
              label=f"Vùng cân bằng = ±{EQUIL_THRESHOLD:.1f}°")
ax_r.axhline(-EQUIL_THRESHOLD, color="green", linestyle=":", linewidth=1.3)
_shade_band(ax_r, t, band_roll)
_mark_peak(ax_r, t_peak_roll, peak_roll_signed)
_vline_collision(ax_r)
_vline_settled(ax_r, recov_roll["t_settled"])
ax_r.set_xlabel("Thời gian (s)", fontsize=10)
ax_r.set_ylabel("Góc (°)", fontsize=10)
ax_r.set_title(
    f"Roll (target = 0°)\nRMSE = {rmse_roll:.2f}°  |  Đỉnh = {peak_roll_signed:.2f}°  |  "
    f"T.cân bằng = {_band_str(band_roll)}  |  T.hồi (va chạm) = {_recov_str(recov_roll)}",
    fontsize=8, fontweight="bold"
)
ax_r.legend(fontsize=8, loc="best")
ax_r.grid(True, alpha=0.35)

# ── Pitch ────────────────────────────────────────────────────────────────────
ax_p = fig2.add_subplot(gs2[0, 1])
ax_p.plot(t, pitch, color="darkorange", linewidth=2, label="Pitch")
ax_p.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_p.axhline( EQUIL_THRESHOLD, color="green", linestyle=":", linewidth=1.3,
              label=f"Vùng cân bằng = ±{EQUIL_THRESHOLD:.1f}°")
ax_p.axhline(-EQUIL_THRESHOLD, color="green", linestyle=":", linewidth=1.3)
_shade_band(ax_p, t, band_pitch)
_mark_peak(ax_p, t_peak_pitch, peak_pitch_signed)
_vline_collision(ax_p)
_vline_settled(ax_p, recov_pitch["t_settled"])
ax_p.set_xlabel("Thời gian (s)", fontsize=10)
ax_p.set_ylabel("Góc (°)", fontsize=10)
ax_p.set_title(
    f"Pitch (target = 0°)\nRMSE = {rmse_pitch:.2f}°  |  Đỉnh = {peak_pitch_signed:.2f}°  |  "
    f"T.cân bằng = {_band_str(band_pitch)}  |  T.hồi (va chạm) = {_recov_str(recov_pitch)}",
    fontsize=8, fontweight="bold"
)
ax_p.legend(fontsize=8, loc="best")
ax_p.grid(True, alpha=0.35)

# ── Yaw ──────────────────────────────────────────────────────────────────────
ax_w = fig2.add_subplot(gs2[0, 2])
ax_w.plot(t, yaw, color="darkred", linewidth=2, label="Yaw")
ax_w.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_w.axhline( EQUIL_THRESHOLD, color="green", linestyle=":", linewidth=1.3,
              label=f"Vùng cân bằng = ±{EQUIL_THRESHOLD:.1f}°")
ax_w.axhline(-EQUIL_THRESHOLD, color="green", linestyle=":", linewidth=1.3)
_shade_band(ax_w, t, band_yaw)
_mark_peak(ax_w, t_peak_yaw, peak_yaw_signed)
_vline_collision(ax_w)
_vline_settled(ax_w, recov_yaw["t_settled"])
ax_w.set_xlabel("Thời gian (s)", fontsize=10)
ax_w.set_ylabel("Góc (°)", fontsize=10)
ax_w.set_title(
    f"Yaw (target = 0°)\nRMSE = {rmse_yaw:.2f}°  |  Đỉnh = {peak_yaw_signed:.2f}°  |  "
    f"T.cân bằng = {_band_str(band_yaw)}  |  T.hồi (va chạm) = {_recov_str(recov_yaw)}",
    fontsize=8, fontweight="bold"
)
ax_w.legend(fontsize=8, loc="best")
ax_w.grid(True, alpha=0.35)

# ── Lưu và hiển thị ──────────────────────────────────────────────────────────
out_path2 = os.path.join(OUTPUT_DIR, "KQ2_quytrinh_goc.png")
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Đã lưu: {out_path2}\n")