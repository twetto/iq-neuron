#!/usr/bin/env python3
"""
Plot IQIF-PC egomotion estimate vs EuRoC ground truth (6-DoF, frame-aligned).

Reads the CSV written by the Rust `euroc_egomotion` example
(frame,ts_ns,vx,vy,vz,wx,wy,wz,gt_vx,gt_vy,gt_vz,qw,qx,qy,qz) and the cam0
`sensor.yaml` (for the body<-camera extrinsic T_BS), rotates the world-frame GT
velocity into the camera frame, derives GT angular rate from the orientation
finite-difference, and draws six subplots (v1 v2 v3, w1 w2 w3): estimate vs GT.

Usage:
    python tests/plot_egomotion.py egomotion_log.csv /path/to/mav0/cam0/sensor.yaml
"""

import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quat_to_R(q):
    """[qw,qx,qy,qz] -> 3x3 rotation R_world_body."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def so3_log(R):
    """Rotation matrix -> axis*angle vector (so(3) log)."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (theta / (2.0 * np.sin(theta)))


def parse_R_bs(sensor_yaml):
    """body<-camera rotation R_bs = top-left 3x3 of the T_BS 4x4 (row-major)."""
    txt = open(sensor_yaml).read()
    block = txt[txt.index("T_BS"):]
    block = block[block.index("data"):]
    nums = [float(v) for v in re.findall(r"[-+0-9.eE]+", block[block.index("["):block.index("]")])]
    T = np.array(nums[:16]).reshape(4, 4)
    return T[:3, :3]


def main():
    if len(sys.argv) < 3:
        print("Usage: plot_egomotion.py <log.csv> <cam0_sensor.yaml>")
        sys.exit(1)
    csv_path, sensor_yaml = sys.argv[1], sys.argv[2]

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        print("empty log")
        sys.exit(1)

    ts = data["ts_ns"].astype(np.float64)
    est = np.stack([data[k] for k in ["vx", "vy", "vz", "wx", "wy", "wz"]], axis=1)
    gt_v_w = np.stack([data["gt_vx"], data["gt_vy"], data["gt_vz"]], axis=1)
    quat = np.stack([data["qw"], data["qx"], data["qy"], data["qz"]], axis=1)

    R_bs = parse_R_bs(sensor_yaml)
    R_sb = R_bs.T  # camera<-body

    n = len(ts)
    gt_cam = np.full((n, 6), np.nan)
    prev = None  # (ts, R_wb)
    for i in range(n):
        if np.any(np.isnan(quat[i])) or np.any(np.isnan(gt_v_w[i])):
            prev = None
            continue
        R_wb = quat_to_R(quat[i])
        gt_cam[i, :3] = R_sb @ (R_wb.T @ gt_v_w[i])         # GT linear velocity, camera frame
        if prev is not None and ts[i] > prev[0]:
            dtg = (ts[i] - prev[0]) * 1e-9
            w_body = so3_log(prev[1].T @ R_wb) / dtg         # GT angular rate, body frame
            gt_cam[i, 3:] = R_sb @ w_body                    # -> camera frame
        prev = (ts[i], R_wb)

    t = (ts - ts[0]) * 1e-9  # seconds from start
    labels = ["v1 (m/s)", "v2 (m/s)", "v3 (m/s)", "w1 (rad/s)", "w2 (rad/s)", "w3 (rad/s)"]

    fig, ax = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("IQIF predictive-coding egomotion vs EuRoC ground truth "
                 "(GT rotated into camera frame)", fontweight="bold")
    for d in range(6):
        ax[d].plot(t, gt_cam[:, d], color="0.5", lw=2.0, label="ground truth")
        ax[d].plot(t, est[:, d], color="tab:red" if d < 3 else "tab:blue", lw=1.0, label="IQIF-PC")
        ax[d].axhline(0, color="0.8", lw=0.6)
        ax[d].set_ylabel(labels[d])
        ax[d].grid(alpha=0.25)
        if d == 0:
            ax[d].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("time (s)")

    # Per-axis RMSE where GT is available.
    valid = ~np.isnan(gt_cam)
    rmse = np.sqrt(np.nanmean((est - gt_cam) ** 2, axis=0))
    print("Per-axis RMSE (est vs GT, camera frame):")
    for d in range(6):
        print(f"  {labels[d]:12s}: {rmse[d]:.4f}  (n={int(valid[:, d].sum())})")

    out = csv_path.rsplit(".", 1)[0] + "_vs_gt.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
