#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  ACT 1 — OpenSim Biomechanical Reference
═══════════════════════════════════════════════════════════════════════
  Presenta el modelo musculoesquelético OpenSim Gait10dof:
  - Esqueleto sagital animado (cadera→rodilla→tobillo→pie)
  - Trayectorias de keypoints apareciendo progresivamente
  - Títulos cinematográficos

  Salida: act1_opensim.mp4  (~60s @ 30fps)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# ── Datos ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'opensim_extraction')
import csv

def load_spatial():
    path = os.path.join(DATA_DIR, 'spatial_trajectories.csv')
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    hip  = np.array([[float(r['hip_x']),  float(r['hip_y'])]  for r in rows])
    knee = np.array([[float(r['knee_x']), float(r['knee_y'])] for r in rows])
    ankle= np.array([[float(r['ankle_x']),float(r['ankle_y'])] for r in rows])
    toe  = np.array([[float(r['toe_x']),  float(r['toe_y'])]  for r in rows])
    pct  = np.array([float(r['gait_pct']) for r in rows])
    return hip, knee, ankle, toe, pct

hip_raw, knee_raw, ankle_raw, toe_raw, gait_pct = load_spatial()

# Resample a 360 puntos para ciclo completo
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

# Añadir pelvis (arriba de la cadera)
pelvis = hip.copy()
pelvis[:, 1] += 200  # 200mm arriba

# Hombro estimado
shoulder = hip.copy()
shoulder[:, 1] += 450
shoulder[:, 0] += 30

# ── Parámetros de video ──────────────────────────────────────────────
FPS = 30
DURATION_S = 55
N_FRAMES = FPS * DURATION_S

OUT_DIR = os.path.dirname(__file__)
OUT_FILE = os.path.join(OUT_DIR, 'act1_opensim.mp4')

# ── Fases (en frames) ───────────────────────────────────────────────
TITLE_IN      = 0
TITLE_HOLD    = int(3.0 * FPS)     # título 3s
SKELETON_IN   = TITLE_HOLD         # esqueleto aparece
WALK_START    = int(5.0 * FPS)     # empieza a caminar
TRAJ_START    = int(12.0 * FPS)    # trayectorias empiezan
LABELS_IN     = int(20.0 * FPS)    # labels aparecen
HIGHLIGHT     = int(30.0 * FPS)    # highlight pie y rodilla
WALK_END      = int(50.0 * FPS)    # cierre
FADE_OUT      = int(52.0 * FPS)    # fade out

# ── Colores premium ─────────────────────────────────────────────────
BG_COLOR      = '#0a0a1a'
BONE_COLOR    = '#c8d0e0'
JOINT_COLOR   = '#ff6b6b'
HIP_COLOR     = '#ffd93d'
KNEE_COLOR    = '#6bcb77'
ANKLE_COLOR   = '#4d96ff'
TOE_COLOR     = '#ff6b9d'
TRAJ_FOOT     = '#ff4444'
TRAJ_KNEE     = '#44ff88'
TITLE_COLOR   = '#e8e8ff'
SUBTITLE_COLOR= '#8888bb'
GRID_COLOR    = '#1a1a3a'

# ── Figura ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_aspect('equal')

# Ajustar limites
all_x = np.concatenate([hip[:,0], knee[:,0], ankle[:,0], toe[:,0], shoulder[:,0]])
all_y = np.concatenate([hip[:,1], knee[:,1], ankle[:,1], toe[:,1], shoulder[:,1]])
pad = 150
ax.set_xlim(all_x.min() - pad*2, all_x.max() + pad*2)
ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Suelo
ground_y = toe[:, 1].min() - 20
ax.axhline(y=ground_y, color='#2a2a4a', linewidth=2, linestyle='-')
ax.fill_between([all_x.min()-pad*2, all_x.max()+pad*2], 
                ground_y-200, ground_y, color='#0d0d25', alpha=0.8)

# Grid sutil
for yg in range(int(all_y.min()-pad), int(all_y.max()+pad), 100):
    ax.axhline(y=yg, color=GRID_COLOR, linewidth=0.3, alpha=0.4)
for xg in range(int(all_x.min()-pad*2), int(all_x.max()+pad*2), 100):
    ax.axvline(x=xg, color=GRID_COLOR, linewidth=0.3, alpha=0.4)

# Elementos animados
bones_line, = ax.plot([], [], '-', color=BONE_COLOR, linewidth=5, 
                       solid_capstyle='round', alpha=0, zorder=5)
bones_thin, = ax.plot([], [], '-', color=BONE_COLOR, linewidth=2, 
                       solid_capstyle='round', alpha=0, zorder=4)

# Joints
joint_scatter = ax.scatter([], [], s=120, c=[JOINT_COLOR], 
                           zorder=10, alpha=0, edgecolors='white', linewidth=1)

# Trayectorias
traj_foot_line, = ax.plot([], [], '-', color=TRAJ_FOOT, linewidth=2.5, alpha=0, zorder=3)
traj_knee_line, = ax.plot([], [], '-', color=TRAJ_KNEE, linewidth=2.5, alpha=0, zorder=3)

# Glow effects para trayectorias
traj_foot_glow, = ax.plot([], [], '-', color=TRAJ_FOOT, linewidth=8, alpha=0, zorder=2)
traj_knee_glow, = ax.plot([], [], '-', color=TRAJ_KNEE, linewidth=8, alpha=0, zorder=2)

# Textos
title_text = ax.text(0.5, 0.92, '', transform=ax.transAxes,
                     fontsize=32, fontweight='bold', color=TITLE_COLOR,
                     ha='center', va='top', fontfamily='sans-serif', alpha=0)
subtitle_text = ax.text(0.5, 0.85, '', transform=ax.transAxes,
                        fontsize=16, color=SUBTITLE_COLOR,
                        ha='center', va='top', fontfamily='sans-serif', alpha=0)
phase_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                     fontsize=11, color='#666688', ha='left', va='bottom',
                     fontfamily='monospace', alpha=0)

# Labels de joints
label_texts = {}
for name, color in [('Hip', HIP_COLOR), ('Knee', KNEE_COLOR), 
                    ('Ankle', ANKLE_COLOR), ('Toe', TOE_COLOR)]:
    label_texts[name] = ax.text(0, 0, f'  {name}', fontsize=11, color=color,
                                fontweight='bold', alpha=0, zorder=15,
                                fontfamily='sans-serif')

# Joint highlights
highlight_circles = {}
for name, color in [('Foot', TRAJ_FOOT), ('Knee', TRAJ_KNEE)]:
    c = Circle((0, 0), 25, fill=False, edgecolor=color, linewidth=3, alpha=0, zorder=12)
    ax.add_patch(c)
    highlight_circles[name] = c

# ── Animación ────────────────────────────────────────────────────────
def ease_in_out(t):
    """Smooth ease in/out, t in [0,1]."""
    return t * t * (3 - 2 * t)

def animate(frame):
    # Progreso del ciclo de marcha
    if frame >= WALK_START:
        walk_frames = frame - WALK_START
        cycle_speed = 1.5  # ciclos por segundo
        phase = (walk_frames / FPS * cycle_speed) % 1.0
        idx = int(phase * N_CYCLE) % N_CYCLE
    else:
        idx = 0

    # ── Título fade in/out ───────────────────────────
    if frame < TITLE_HOLD:
        alpha_t = ease_in_out(min(1.0, frame / (FPS * 1.5)))
        title_text.set_text('OpenSim Gait Analysis')
        title_text.set_alpha(alpha_t)
        subtitle_text.set_text('Rajagopal 10-DOF Musculoskeletal Model  ·  Sagittal Plane')
        subtitle_text.set_alpha(alpha_t * 0.7)
    elif frame < SKELETON_IN + FPS:
        fade = 1.0 - ease_in_out((frame - TITLE_HOLD) / FPS)
        title_text.set_alpha(fade)
        subtitle_text.set_alpha(fade * 0.7)
    else:
        title_text.set_alpha(0)
        subtitle_text.set_alpha(0)

    # ── Esqueleto fade in ────────────────────────────
    if frame >= SKELETON_IN:
        skel_alpha = ease_in_out(min(1.0, (frame - SKELETON_IN) / (FPS * 2)))
    else:
        skel_alpha = 0

    # Posiciones actuales
    h, k, a, t = hip[idx], knee[idx], ankle[idx], toe[idx]
    p = pelvis[idx]
    s = shoulder[idx]

    # Huesos principales (femur, tibia, pie)
    bones_line.set_data([h[0], k[0], a[0], t[0]], 
                        [h[1], k[1], a[1], t[1]])
    bones_line.set_alpha(skel_alpha)

    # Torso
    bones_thin.set_data([h[0], p[0], s[0]], [h[1], p[1], s[1]])
    bones_thin.set_alpha(skel_alpha * 0.5)

    # Joints
    joint_positions = np.array([h, k, a, t])
    joint_scatter.set_offsets(joint_positions)
    joint_scatter.set_alpha(skel_alpha)
    colors = [HIP_COLOR, KNEE_COLOR, ANKLE_COLOR, TOE_COLOR]
    joint_scatter.set_facecolors(colors)

    # ── Trayectorias progresivas ─────────────────────
    if frame >= TRAJ_START:
        traj_progress = min(1.0, (frame - TRAJ_START) / (FPS * 6))
        n_pts = max(2, int(traj_progress * N_CYCLE))
        traj_alpha = ease_in_out(min(1.0, (frame - TRAJ_START) / (FPS * 2))) * 0.85

        traj_foot_line.set_data(toe[:n_pts, 0], toe[:n_pts, 1])
        traj_foot_line.set_alpha(traj_alpha)
        traj_foot_glow.set_data(toe[:n_pts, 0], toe[:n_pts, 1])
        traj_foot_glow.set_alpha(traj_alpha * 0.15)

        traj_knee_line.set_data(knee[:n_pts, 0], knee[:n_pts, 1])
        traj_knee_line.set_alpha(traj_alpha)
        traj_knee_glow.set_data(knee[:n_pts, 0], knee[:n_pts, 1])
        traj_knee_glow.set_alpha(traj_alpha * 0.15)

    # ── Labels ───────────────────────────────────────
    if frame >= LABELS_IN:
        lbl_alpha = ease_in_out(min(1.0, (frame - LABELS_IN) / (FPS * 1.5)))
        for name, pos in [('Hip', h), ('Knee', k), ('Ankle', a), ('Toe', t)]:
            label_texts[name].set_position((pos[0] + 15, pos[1] + 15))
            label_texts[name].set_alpha(lbl_alpha)

    # ── Highlights ───────────────────────────────────
    if frame >= HIGHLIGHT:
        hl_alpha = ease_in_out(min(1.0, (frame - HIGHLIGHT) / FPS))
        pulse = 0.5 + 0.5 * np.sin(frame * 0.15)
        
        highlight_circles['Foot'].center = (t[0], t[1])
        highlight_circles['Foot'].set_alpha(hl_alpha * pulse)
        highlight_circles['Foot'].set_radius(20 + 10 * pulse)
        
        highlight_circles['Knee'].center = (k[0], k[1])
        highlight_circles['Knee'].set_alpha(hl_alpha * pulse)
        highlight_circles['Knee'].set_radius(20 + 10 * pulse)

    # ── Phase indicator ──────────────────────────────
    if frame >= WALK_START:
        gait_phase = ((frame - WALK_START) / FPS * 1.5) % 1.0 * 100
        phase_text.set_text(f'Gait Cycle: {gait_phase:5.1f}%  |  Frame {frame}/{N_FRAMES}')
        phase_text.set_alpha(0.6)

    # ── Fade out ─────────────────────────────────────
    if frame >= FADE_OUT:
        fade = 1.0 - ease_in_out(min(1.0, (frame - FADE_OUT) / (FPS * 3)))
        for artist in [bones_line, bones_thin, traj_foot_line, traj_knee_line,
                       traj_foot_glow, traj_knee_glow, phase_text]:
            artist.set_alpha(artist.get_alpha() * fade)
        joint_scatter.set_alpha(joint_scatter.get_alpha() * fade)
        for lbl in label_texts.values():
            lbl.set_alpha(lbl.get_alpha() * fade)
        for c in highlight_circles.values():
            c.set_alpha(c.get_alpha() * fade)

    return []

print(f"Rendering Act 1: OpenSim ({N_FRAMES} frames @ {FPS}fps = {DURATION_S}s)")
print(f"Output: {OUT_FILE}")

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
anim.save(OUT_FILE, writer=writer)
print(f"✓ Act 1 saved: {OUT_FILE}")
plt.close()
