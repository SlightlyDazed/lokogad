#!/usr/bin/env python3
"""
shape_fit.py — Trajectory-shape and position errors of the stored mechanisms.

Each mechanism trajectory (foot, knee) is fitted onto its human reference the way the eye
"squares up" two curves:
  * centred on the reference centroid and scaled to its RMS size;
  * rotated by at most +/-30 deg (the device may be tilted, not turned upside down);
  * shifted in phase (360 offsets, either crank direction).
The mirror image of the mechanism (x -> -x, i.e. the device seen from the other side) is
chosen once per mechanism and applied to all its trajectories.

  shape error    = RMS distance after fitting / span of the reference
  position error = |centroid_mech - centroid_ref| / |centroid_ref|   (hip = crank axis)

References: Fourier-smoothed gait (H = 7), opensim_extraction/*_fourier_tensor.pt
(use --raw for the interpolated OpenSim samples, *_ref_tensor.pt).

Usage (from the repository root):
    python analysis/shape_fit.py [--raw] [--fig results/shape_fit.png]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import config_jansen as cj  # noqa: E402
from general_solver import compute_full_trajectory  # noqa: E402
from linkage_genome import LinkageGenome  # noqa: E402

MAX_ROT_DEG = 30.0
W_FOOT, W_KNEE = 0.6, 0.4


def load_refs(raw=False):
    if raw:
        return cj.load_reference_trajectories()
    d = os.path.join(ROOT, "opensim_extraction")

    def ld(j):
        return torch.load(os.path.join(d, f"{j}_fourier_tensor.pt"), map_location="cpu",
                          weights_only=True).numpy()

    hip = ld("hip")
    hip_xz = np.column_stack([hip[:, 0], hip[:, 2]]) * 1000
    return {j: np.column_stack([ld(j)[:, 0], ld(j)[:, 2]]) * 1000 - hip_xz for j in ("foot", "knee")}


def fit_curve(m, r, max_rot_deg=MAX_ROT_DEG):
    """Fit curve m (N,2) onto reference r (N,2). Returns errors and the fitted curve."""
    r_mean = r.mean(0)
    rc = r - r_mean
    mc = m - m.mean(0)
    sr = math.sqrt((rc ** 2).sum(1).mean())
    sm = math.sqrt((mc ** 2).sum(1).mean())
    mc = mc * sr / sm
    span = max(np.ptp(rc[:, 0]), np.ptp(rc[:, 1]))
    lim = math.radians(max_rot_deg)
    best = None
    for direction in (1, -1):
        md = mc[::direction]
        for tau in range(len(md)):
            P = np.roll(md, tau, 0)
            a = np.sum(P[:, 0] * rc[:, 0] + P[:, 1] * rc[:, 1])
            b = np.sum(P[:, 0] * rc[:, 1] - P[:, 1] * rc[:, 0])
            th = max(-lim, min(lim, math.atan2(b, a)))   # closed-form optimum, clamped
            c, s = math.cos(th), math.sin(th)
            A = P @ np.array([[c, s], [-s, c]])
            rmsd = math.sqrt(np.mean(np.sum((A - rc) ** 2, 1)))
            if best is None or rmsd < best[0]:
                best = (rmsd, A + r_mean, math.degrees(th))
    return {"shape": best[0] / span, "size_ratio": sm / sr, "rotation_deg": best[2], "fitted": best[1]}


def pos_error(m, r):
    return float(np.linalg.norm(m.mean(0) - r.mean(0)) / np.linalg.norm(r.mean(0)))


def evaluate(foot, knee, ref):
    out = None
    for mirror in (False, True):
        s = np.array([-1.0, 1.0]) if mirror else np.array([1.0, 1.0])
        f, k = fit_curve(foot * s, ref["foot"]), fit_curve(knee * s, ref["knee"])
        score = W_FOOT * f["shape"] + W_KNEE * k["shape"]
        if out is None or score < out["score"]:
            out = {"score": score, "mirror": mirror, "foot": f, "knee": k,
                   "pos_foot": pos_error(foot * s, ref["foot"]), "pos_knee": pos_error(knee * s, ref["knee"])}
    return out


def mechanisms():
    res = os.path.join(ROOT, "results")
    bm = json.load(open(os.path.join(res, "best_mechanism.json")))
    tj = cj.compute_trajectories(np.array(bm["magic_numbers"]))
    gd = json.load(open(os.path.join(res, "best_genome.json")))
    g = LinkageGenome.from_dict(gd)
    T = compute_full_trajectory(g.nodes, g.edges, 360)
    return [("Jansen topology (NSGA-II genome)", tj[cj.POINT_FOOT], tj[cj.POINT_KNEE]),
            ("12-link mechanism (topological search)", T[gd["foot_id"]], T[gd["knee_id"]])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="use the raw interpolated OpenSim references")
    ap.add_argument("--fig", default=None, help="save a figure of the fitted curves")
    args = ap.parse_args()
    ref = load_refs(args.raw)
    rows = []
    for name, foot, knee in mechanisms():
        e = evaluate(foot, knee, ref)
        rows.append((name, e))
        print(f"{name}\n  mirrored: {e['mirror']}\n"
              f"  foot shape {100*e['foot']['shape']:.1f}%  (rotation {e['foot']['rotation_deg']:.1f} deg, size {e['foot']['size_ratio']:.2f})\n"
              f"  knee shape {100*e['knee']['shape']:.1f}%  (rotation {e['knee']['rotation_deg']:.1f} deg, size {e['knee']['size_ratio']:.2f})\n"
              f"  position   foot {100*e['pos_foot']:.1f}%  knee {100*e['pos_knee']:.1f}%")
    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
        for col, (name, e) in enumerate(rows):
            for row, j in enumerate(("foot", "knee")):
                ax = axes[row, col]
                r, fit = ref[j], e[j]
                ax.plot(r[:, 0], r[:, 1], color="0.15", lw=2.2, ls="--", label="Human reference")
                ax.plot(fit["fitted"][:, 0], fit["fitted"][:, 1], lw=1.8,
                        color="#c0392b" if j == "foot" else "#2471a3",
                        label=f"Mechanism, fitted ({100*fit['shape']:.1f}%)")
                ax.set_title(f"{j.capitalize()} - {name}\n{'mirrored' if e['mirror'] else 'not mirrored'}, "
                             f"size ratio {fit['size_ratio']:.2f}, rotation {fit['rotation_deg']:.1f}°", fontsize=8.5)
                ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7.5)
                ax.set_xlabel("Forward position (mm)"); ax.set_ylabel("Vertical position (mm)")
        fig.tight_layout(); fig.savefig(args.fig, dpi=300)
        print("figure:", args.fig)


if __name__ == "__main__":
    main()
