#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  ACT 2 — Trajectory Extraction
═══════════════════════════════════════════════════════════════════════
  Muestra cómo se extraen las trayectorias del modelo OpenSim:
  - Panel izquierdo: esqueleto sagital (simplificado)
  - Panel derecho: curvas de trayectoria pie y rodilla apareciendo
  - Flechas de extracción entre paneles
  - Tabla de datos apareciendo progresivamente

  Salida: act2_extraction.mp4  (~55s @ 30fps)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from config_jansen import load_reference_trajectories, GAIT_DIR

# ── Datos ────────────────────────────────────────────────────────────
import csv
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'opensim_extraction')

def load_spatial():
    path = os.path.join(DATA_DIR, 'spatial_trajectories.csv')
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    hip  = np.array([[float(r['hip_x']),  float(r['hip_y'])]  for r in rows])
    knee = np.array([[float(r['knee_x']), float(r['knee_y'])] for r in rows])
    ankle= np.array([[float(r['ankle_x']),float(r['ankle_y'])] for r in rows])
    toe  = np.array([[float(r['toe_x']),  float(r['toe_y'])]  for r in rows])
    return hip, knee, ankle, toe

hip_raw, knee_raw, ankle_raw, toe_raw = load_spatial()

N_CYCLE = 360
t_orig = np.linspace(0, 1, len(hip_raw))
t_new  = np.linspace(0, 1, N_CYCLE)
hip  = np.column_stack([np.interp(t_new, t_orig, hip_raw[:,0]),
                        np.interp(t_new, t_orig, hip_raw[:,1])])
knee = np.column_stack([np.interp(t_new, t_orig, knee_raw[:,0]),
                        np.interp(t_new, t_orig, knee_raw[:,1])])
ankle= np.column_stack([np.interp(t_new, t_orig, ankle_raw[:,0]),
                        np.interp(t_new, t_orig, ankle_raw[:,1])])
toe  = np.column_stack([np.interp(t_new, t_orig, toe_raw[:,0]),
                        np.interp(t_new, t_orig, toe_raw[:,1])])

# Cargar referencia procesada
ref = load_reference_trajectories()
ref_foot = ref['foot']  # (360, 2) relativo a cadera, mm
ref_knee = ref['knee']

# ── Parámetros de video ──────────────────────────────────────────────
FPS = 30
DURATION_S = 55
N_FRAMES = FPS * DURATION_S
OUT_DIR = os.path.dirname(__file__)
OUT_FILE = os.path.join(OUT_DIR, 'act2_extraction.mp4')

# ── Colores ──────────────────────────────────────────────────────────
BG = '#0a0a1a'
BONE   = '#c8d0e0'
RED    = '#ff4444'
GREEN  = '#44ff88'
BLUE   = '#4d96ff'
YELLOW = '#ffd93d'
GRAY   = '#666688'
CYAN   = '#00e5ff'

# ── Fases ────────────────────────────────────────────────────────────
TITLE_END     = int(3.0 * FPS)
SKEL_WALK     = int(4.0 * FPS)
EXTRACT_START = int(10.0 * FPS)
CURVES_START  = int(15.0 * FPS)
CURVES_FULL   = int(30.0 * FPS)
DATA_START    = int(35.0 * FPS)
FINAL_HOLD    = int(48.0 * FPS)
FADE_OUT      = int(51.0 * FPS)

# ── Figura con 2 paneles ────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(BG)

ax_skel = fig.add_axes([0.03, 0.08, 0.38, 0.82])   # izq: esqueleto
ax_traj = fig.add_axes([0.50, 0.08, 0.47, 0.82])    # der: trayectorias

for ax in [ax_skel, ax_traj]:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color('#2a2a4a')
        spine.set_linewidth(0.5)

# ── Panel izq: esqueleto ─────────────────────────────────────────────
all_x = np.concatenate([hip[:,0], knee[:,0], ankle[:,0], toe[:,0]])
all_y = np.concatenate([hip[:,1], knee[:,1], ankle[:,1], toe[:,1]])
pad = 120
ax_skel.set_xlim(all_x.min()-pad, all_x.max()+pad)
ax_skel.set_ylim(all_y.min()-pad, all_y.max()+pad)
ax_skel.set_xticks([]); ax_skel.set_yticks([])

skel_title = ax_skel.text(0.5, 1.03, 'Sagittal Skeleton', transform=ax_skel.transAxes,
                           fontsize=14, color=BONE, ha='center', fontweight='bold', alpha=0)

ground_y = toe[:,1].min() - 15
ax_skel.axhline(y=ground_y, color='#2a2a4a', linewidth=1.5)

skel_bones, = ax_skel.plot([], [], '-', color=BONE, linewidth=5,
                            solid_capstyle='round', alpha=0, zorder=5)
skel_joints = ax_skel.scatter([], [], s=100, zorder=10, alpha=0, 
                               edgecolors='white', linewidth=1)
skel_toe_trail, = ax_skel.plot([], [], '-', color=RED, linewidth=2, alpha=0, zorder=3)
skel_knee_trail, = ax_skel.plot([], [], '-', color=GREEN, linewidth=2, alpha=0, zorder=3)

# Dot markers en el esqueleto (toe y knee pulsantes)
skel_toe_dot = ax_skel.scatter([], [], s=200, c=[RED], zorder=15, alpha=0)
skel_knee_dot = ax_skel.scatter([], [], s=200, c=[GREEN], zorder=15, alpha=0)

# ── Panel der: trayectorias ──────────────────────────────────────────
ax_traj.set_xlim(ref_foot[:,0].min()-50, ref_foot[:,0].max()+50)
ax_traj.set_ylim(min(ref_foot[:,1].min(), ref_knee[:,1].min())-80,
                 max(ref_foot[:,1].max(), ref_knee[:,1].max())+80)
ax_traj.set_xlabel('X (mm)', color=GRAY, fontsize=10)
ax_traj.set_ylabel('Y (mm)', color=GRAY, fontsize=10)
ax_traj.tick_params(colors=GRAY, labelsize=8)
ax_traj.grid(True, color='#1a1a3a', linewidth=0.3, alpha=0.6)

traj_title = ax_traj.text(0.5, 1.03, 'Extracted Trajectories (Sagittal Plane)',
                           transform=ax_traj.transAxes, fontsize=14, color=CYAN,
                           ha='center', fontweight='bold', alpha=0)

# Trayectorias objetivo
foot_curve, = ax_traj.plot([], [], '-', color=RED, linewidth=2.5, alpha=0, zorder=5,
                            label='Toe trajectory')
foot_glow, = ax_traj.plot([], [], '-', color=RED, linewidth=8, alpha=0, zorder=3)
knee_curve, = ax_traj.plot([], [], '-', color=GREEN, linewidth=2.5, alpha=0, zorder=5,
                            label='Knee trajectory')
knee_glow, = ax_traj.plot([], [], '-', color=GREEN, linewidth=8, alpha=0, zorder=3)

# Dot actual en curva
foot_dot = ax_traj.scatter([], [], s=120, c=[RED], zorder=15, alpha=0, edgecolors='white', linewidth=1)
knee_dot = ax_traj.scatter([], [], s=120, c=[GREEN], zorder=15, alpha=0, edgecolors='white', linewidth=1)

legend = ax_traj.legend(loc='upper right', fontsize=10, facecolor=BG, edgecolor='#333355',
                         labelcolor='white', framealpha=0.8)
legend.set_visible(False)

# ── Título principal ─────────────────────────────────────────────────
main_title = fig.text(0.5, 0.97, '', fontsize=28, color='#e8e8ff',
                       ha='center', va='top', fontweight='bold', alpha=0)
main_sub = fig.text(0.5, 0.935, '', fontsize=14, color=GRAY,
                     ha='center', va='top', alpha=0)

# Arrow annotation (between panels)
arrow_text = fig.text(0.44, 0.5, '', fontsize=40, color=CYAN, 
                       ha='center', va='center', alpha=0, fontweight='bold')

# Data info text
data_lines = [
    'Source: OpenSim Gait10dof18musc',
    'Keypoints: Hip · Knee · Ankle · Toe',
    'Plane: Sagittal (X, Z)',
    'Reference frame: Hip-relative',
    f'Points per cycle: {N_CYCLE}',
    f'Foot stride: {ref_foot[:,0].max()-ref_foot[:,0].min():.0f} mm',
    f'Knee range Y: {ref_knee[:,1].max()-ref_knee[:,1].min():.0f} mm',
]
data_text_objs = []
for i, line in enumerate(data_lines):
    t = fig.text(0.52, 0.32 - i*0.032, line, fontsize=10, color='#9999bb',
                 fontfamily='monospace', alpha=0)
    data_text_objs.append(t)

# ── Animación ────────────────────────────────────────────────────────
def ease(t):
    return t * t * (3 - 2*t)

trail_foot_x, trail_foot_y = [], []
trail_knee_x, trail_knee_y = [], []

def animate(frame):
    # Gait cycle
    if frame >= SKEL_WALK:
        phase = ((frame - SKEL_WALK) / FPS * 1.2) % 1.0
        idx = int(phase * N_CYCLE) % N_CYCLE
    else:
        idx = 0

    h, k, a, t = hip[idx], knee[idx], ankle[idx], toe[idx]

    # ── Título ───────────────────────────────────────
    if frame < TITLE_END:
        a_t = ease(min(1.0, frame / (FPS * 1.5)))
        main_title.set_text('Trajectory Extraction')
        main_title.set_alpha(a_t)
        main_sub.set_text('From OpenSim markers to optimization targets')
        main_sub.set_alpha(a_t * 0.7)
    elif frame < TITLE_END + FPS:
        fade = 1.0 - ease((frame - TITLE_END) / FPS)
        main_title.set_alpha(fade)
        main_sub.set_alpha(fade * 0.7)
    else:
        main_title.set_alpha(0)
        main_sub.set_alpha(0)

    # ── Esqueleto ────────────────────────────────────
    sk_alpha = ease(min(1.0, max(0, (frame - SKEL_WALK + FPS) / (FPS * 1.5))))
    skel_bones.set_data([h[0],k[0],a[0],t[0]], [h[1],k[1],a[1],t[1]])
    skel_bones.set_alpha(sk_alpha)
    
    jp = np.array([h, k, a, t])
    skel_joints.set_offsets(jp)
    skel_joints.set_facecolors([YELLOW, GREEN, BLUE, RED])
    skel_joints.set_alpha(sk_alpha)
    skel_title.set_alpha(sk_alpha)

    # ── Trails en esqueleto ──────────────────────────
    if frame >= EXTRACT_START and frame >= SKEL_WALK:
        trail_alpha = ease(min(1.0, (frame - EXTRACT_START) / (FPS * 2)))
        
        trail_foot_x.append(t[0])
        trail_foot_y.append(t[1])
        trail_knee_x.append(k[0])
        trail_knee_y.append(k[1])
        
        # Solo últimos N_CYCLE puntos
        max_pts = N_CYCLE
        skel_toe_trail.set_data(trail_foot_x[-max_pts:], trail_foot_y[-max_pts:])
        skel_toe_trail.set_alpha(trail_alpha * 0.6)
        skel_knee_trail.set_data(trail_knee_x[-max_pts:], trail_knee_y[-max_pts:])
        skel_knee_trail.set_alpha(trail_alpha * 0.6)

    # ── Puntos pulsantes ─────────────────────────────
    if frame >= EXTRACT_START:
        pulse = 0.5 + 0.5 * np.sin(frame * 0.2)
        ext_alpha = ease(min(1.0, (frame - EXTRACT_START) / FPS))
        
        skel_toe_dot.set_offsets([[t[0], t[1]]])
        skel_toe_dot.set_alpha(ext_alpha * pulse)
        skel_knee_dot.set_offsets([[k[0], k[1]]])
        skel_knee_dot.set_alpha(ext_alpha * pulse)
        
        # Arrow
        arrow_text.set_text('→')
        arrow_text.set_alpha(ext_alpha * (0.5 + 0.5 * pulse))

    # ── Curvas de trayectoria (panel derecho) ────────
    if frame >= CURVES_START:
        prog = min(1.0, (frame - CURVES_START) / (CURVES_FULL - CURVES_START))
        n_pts = max(2, int(prog * N_CYCLE))
        c_alpha = ease(min(1.0, (frame - CURVES_START) / (FPS * 2)))

        foot_curve.set_data(ref_foot[:n_pts, 0], ref_foot[:n_pts, 1])
        foot_curve.set_alpha(c_alpha)
        foot_glow.set_data(ref_foot[:n_pts, 0], ref_foot[:n_pts, 1])
        foot_glow.set_alpha(c_alpha * 0.12)

        knee_curve.set_data(ref_knee[:n_pts, 0], ref_knee[:n_pts, 1])
        knee_curve.set_alpha(c_alpha)
        knee_glow.set_data(ref_knee[:n_pts, 0], ref_knee[:n_pts, 1])
        knee_glow.set_alpha(c_alpha * 0.12)

        # Cursor dots
        ci = min(n_pts - 1, idx % n_pts)
        foot_dot.set_offsets([[ref_foot[ci, 0], ref_foot[ci, 1]]])
        foot_dot.set_alpha(c_alpha)
        knee_dot.set_offsets([[ref_knee[ci, 0], ref_knee[ci, 1]]])
        knee_dot.set_alpha(c_alpha)

        traj_title.set_alpha(c_alpha)
        if not legend.get_visible() and prog > 0.5:
            legend.set_visible(True)

    # ── Data info ────────────────────────────────────
    if frame >= DATA_START:
        for i, txt in enumerate(data_text_objs):
            delay = i * int(FPS * 0.5)
            local_frame = frame - DATA_START - delay
            if local_frame > 0:
                txt.set_alpha(ease(min(1.0, local_frame / (FPS * 0.8))))

    # ── Fade out ─────────────────────────────────────
    if frame >= FADE_OUT:
        fade = 1.0 - ease(min(1.0, (frame - FADE_OUT) / (FPS * 3)))
        for artist in [skel_bones, skel_toe_trail, skel_knee_trail,
                        foot_curve, foot_glow, knee_curve, knee_glow,
                        arrow_text, skel_title, traj_title]:
            artist.set_alpha(artist.get_alpha() * fade)
        for s in [skel_joints, skel_toe_dot, skel_knee_dot, foot_dot, knee_dot]:
            s.set_alpha(s.get_alpha() * fade)
        for txt in data_text_objs:
            txt.set_alpha(txt.get_alpha() * fade)
        if legend.get_visible():
            for text in legend.get_texts():
                text.set_alpha(fade)

    return []

print(f"Rendering Act 2: Trajectory Extraction ({N_FRAMES} frames @ {FPS}fps = {DURATION_S}s)")
print(f"Output: {OUT_FILE}")

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
anim.save(OUT_FILE, writer=writer)
print(f"✓ Act 2 saved: {OUT_FILE}")
plt.close()
