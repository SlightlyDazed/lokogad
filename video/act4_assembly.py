#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  ACT 4 — Assembly & Motion (URDF)
═══════════════════════════════════════════════════════════════════════
  Muestra el ensamblaje pieza por pieza del mecanismo final:
  - Fase 1: Barras aparecen una por una con nombre
  - Fase 2: Pines/pivotes aparecen conectando
  - Fase 3: Mecanismo empieza a moverse lentamente
  - Fase 4: Velocidad completa con trazas

  Basado en la lógica de animate_evolved.py --thick

  Salida: act4_assembly.mp4  (~60s @ 30fps)
"""
import os, sys, json, math, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import load_reference_trajectories, RESULT_DIR

# ── Datos ────────────────────────────────────────────────────────────
genome_path = os.path.join(RESULT_DIR, 'best_genome.json')
with open(genome_path) as f:
    data = json.load(f)
g = LinkageGenome.from_dict(data)
ref = load_reference_trajectories()

trajs = compute_full_trajectory(g.nodes, g.edges)
foot_id = g.foot_id or 7
knee_id = g.knee_id or 6
foot_traj = trajs[foot_id]

# ── Parámetros de video ──────────────────────────────────────────────
FPS = 30
DURATION_S = 60
N_FRAMES = FPS * DURATION_S
OUT_DIR = os.path.dirname(__file__)
OUT_FILE = os.path.join(OUT_DIR, 'act4_assembly.mp4')

# ── Colores ──────────────────────────────────────────────────────────
BG = '#0a0a1a'
COLORS = ['#ff6b6b','#6bcb77','#4d96ff','#ffd93d','#cc65fe',
          '#45b7d1','#ff9ff3','#ff6348','#7bed9f','#70a1ff',
          '#eccc68','#a29bfe']
WHITE  = '#e8e8ff'
GRAY   = '#666688'
CYAN   = '#00e5ff'
RED    = '#ff4444'
GREEN  = '#44ff88'

# ── Fases ────────────────────────────────────────────────────────────
TITLE_END     = int(3.0 * FPS)
ASSEMBLE_START = int(4.0 * FPS)
# Cada barra aparece durante ~1.2s
BAR_DURATION  = int(1.2 * FPS)
N_BARS = len(g.edges)
PINS_START    = ASSEMBLE_START + N_BARS * BAR_DURATION + int(1.0 * FPS)
PINS_END      = PINS_START + int(3.0 * FPS)
MOTION_START  = PINS_END + int(1.0 * FPS)
FULL_SPEED    = MOTION_START + int(4.0 * FPS)
FADE_OUT      = N_FRAMES - int(4.0 * FPS)

# ── Resolver posiciones para el ensamblaje (ángulo 0) ────────────────
positions_0 = solve_linkage(g.nodes, g.edges, 0)

# Offset para centrar + ref
mech_flipped = np.column_stack([-foot_traj[:, 0], foot_traj[:, 1]])
offset = ref['foot'].mean(axis=0) - mech_flipped.mean(axis=0)
OX, OY = offset[0], offset[1]

def transform_pos(pos):
    """Transform nodo position to display coords."""
    return (-pos[0] + OX, pos[1] + OY)

# ── Figura ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_aspect('equal')

# Calcular límites
all_pts = []
for nid, pos in positions_0.items():
    all_pts.append(transform_pos(pos))
all_pts = np.array(all_pts)
cx, cy = all_pts.mean(axis=0)
rng = max(all_pts[:,0].max()-all_pts[:,0].min(), all_pts[:,1].max()-all_pts[:,1].min())
margin = rng * 0.35
ax.set_xlim(cx - rng/2 - margin, cx + rng/2 + margin)
ax.set_ylim(cy - rng/2 - margin, cy + rng/2 + margin)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Grid sutil
for i in range(50):
    xg = cx - rng/2 - margin + i * (rng + 2*margin) / 50
    ax.axvline(x=xg, color='#1a1a3a', linewidth=0.2, alpha=0.3)
    yg = cy - rng/2 - margin + i * (rng + 2*margin) / 50
    ax.axhline(y=yg, color='#1a1a3a', linewidth=0.2, alpha=0.3)

# Suelo
ground_y = min(transform_pos(positions_0[nid])[1] for nid in positions_0) - rng * 0.1
ax.axhline(y=ground_y, color='#2a2a4a', linewidth=2)

# ── Elementos de dibujo ─────────────────────────────────────────────
bar_lines = []
bar_label_texts = []
for i, e in enumerate(g.edges):
    line, = ax.plot([], [], '-', color=COLORS[i % len(COLORS)], 
                     linewidth=6, solid_capstyle='round', alpha=0, zorder=5)
    bar_lines.append(line)
    
    # Nombre de la barra
    n1, n2 = e['n1'], e['n2']
    p1 = transform_pos(positions_0[n1])
    p2 = transform_pos(positions_0[n2])
    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    
    lbl = ax.text(mx, my + 20, f'Bar {i}', fontsize=8, color=COLORS[i % len(COLORS)],
                   ha='center', fontweight='bold', alpha=0, zorder=20,
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, 
                            edgecolor=COLORS[i % len(COLORS)], alpha=0.8))
    bar_label_texts.append(lbl)

# Pines (joints)
node_scatter = ax.scatter([], [], s=150, zorder=10, alpha=0, 
                           edgecolors='white', linewidth=1.5)

# Trazas del pie
foot_trace, = ax.plot([], [], '-', color=RED, linewidth=2, alpha=0, zorder=3)
foot_trace_glow, = ax.plot([], [], '-', color=RED, linewidth=7, alpha=0, zorder=2)

# Referencia
ref_foot_line, = ax.plot([], [], '--', color=CYAN, linewidth=2, alpha=0, zorder=2)

# Textos
main_title = fig.text(0.5, 0.96, '', fontsize=28, color=WHITE,
                       ha='center', va='top', fontweight='bold', alpha=0)
main_sub = fig.text(0.5, 0.915, '', fontsize=14, color=GRAY,
                     ha='center', va='top', alpha=0)

status_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=14,
                       color=CYAN, fontfamily='monospace', fontweight='bold',
                       alpha=0, va='top')

speed_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, fontsize=11,
                      color=GRAY, fontfamily='monospace', ha='right', alpha=0)

# ── Referencia pie plot ──────────────────────────────────────────────
ref_x = ref['foot'][:, 0]
ref_y = ref['foot'][:, 1]

# ── Animación ────────────────────────────────────────────────────────
def ease(t):
    return t * t * (3 - 2*t)

foot_trail_x, foot_trail_y = [], []

def animate(frame):
    # ── Título ───────────────────────────────────────
    if frame < TITLE_END:
        a_t = ease(min(1.0, frame / (FPS * 1.5)))
        main_title.set_text('Mechanism Assembly & Motion')
        main_title.set_alpha(a_t)
        main_sub.set_text('Piece-by-piece construction of the optimized linkage')
        main_sub.set_alpha(a_t * 0.7)
    elif frame < TITLE_END + int(FPS * 1.5):
        fade = 1.0 - ease((frame - TITLE_END) / (FPS * 1.5))
        main_title.set_alpha(fade)
        main_sub.set_alpha(fade * 0.7)
    else:
        main_title.set_alpha(0)
        main_sub.set_alpha(0)

    # ── Determinar ángulo ────────────────────────────
    if frame >= MOTION_START:
        motion_t = frame - MOTION_START
        # Velocidad progresiva
        if frame < FULL_SPEED:
            speed = 5.0 + 25.0 * ease((frame - MOTION_START) / (FULL_SPEED - MOTION_START))
        else:
            speed = 30.0
        angle = (motion_t / FPS * speed) % 360
    else:
        angle = 0

    positions = solve_linkage(g.nodes, g.edges, angle)
    if positions is None:
        return []

    # ── Ensamblaje: barras una por una ───────────────
    for i, e in enumerate(g.edges):
        bar_start_frame = ASSEMBLE_START + i * BAR_DURATION
        
        if frame >= bar_start_frame:
            # Alpha de aparición
            appear_progress = min(1.0, (frame - bar_start_frame) / (FPS * 0.8))
            bar_alpha = ease(appear_progress)
            
            n1, n2 = e['n1'], e['n2']
            p1 = transform_pos(positions[n1])
            p2 = transform_pos(positions[n2])
            
            bar_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            bar_lines[i].set_alpha(bar_alpha)
            
            # Label aparece y luego se desvanece
            label_duration = FPS * 3
            if (frame - bar_start_frame) < label_duration:
                label_alpha = bar_alpha if (frame - bar_start_frame) < FPS else \
                              max(0, 1.0 - (frame - bar_start_frame - FPS) / (label_duration - FPS))
                mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
                bar_label_texts[i].set_position((mx, my + 20))
                bar_label_texts[i].set_alpha(label_alpha * 0.8)
            else:
                bar_label_texts[i].set_alpha(0)

    # ── Pines ────────────────────────────────────────
    if frame >= PINS_START:
        pin_alpha = ease(min(1.0, (frame - PINS_START) / (FPS * 1.5)))
        
        pin_positions = []
        pin_colors = []
        for n in g.nodes:
            nid = n['id']
            if nid in positions:
                p = transform_pos(positions[nid])
                pin_positions.append(p)
                if n['type'] == 'ground':
                    pin_colors.append('#ffffff')
                elif n['type'] == 'crank':
                    pin_colors.append('#ff4444')
                elif n['type'] == 'output_foot':
                    pin_colors.append('#4d96ff')
                elif n['type'] == 'output_knee':
                    pin_colors.append('#44ff88')
                else:
                    pin_colors.append('#ffd93d')
        
        if pin_positions:
            node_scatter.set_offsets(pin_positions)
            node_scatter.set_facecolors(pin_colors)
            node_scatter.set_alpha(pin_alpha)

    # ── Status text ──────────────────────────────────
    if frame >= ASSEMBLE_START and frame < PINS_START:
        bar_idx = min(N_BARS - 1, (frame - ASSEMBLE_START) // BAR_DURATION)
        status_text.set_text(f'Assembling: Bar {bar_idx + 1}/{N_BARS}')
        status_text.set_alpha(0.9)
    elif frame >= PINS_START and frame < MOTION_START:
        status_text.set_text('Connecting Joints...')
        status_text.set_alpha(0.9)
    elif frame >= MOTION_START:
        status_text.set_text('▶ RUNNING')
        pulse = 0.6 + 0.4 * np.sin(frame * 0.1)
        status_text.set_alpha(pulse)

    # ── Trazas del pie durante movimiento ────────────
    if frame >= MOTION_START:
        foot_pos = positions.get(foot_id)
        if foot_pos:
            fp = transform_pos(foot_pos)
            foot_trail_x.append(fp[0])
            foot_trail_y.append(fp[1])

            trail_alpha = ease(min(1.0, (frame - MOTION_START) / (FPS * 3)))
            foot_trace.set_data(foot_trail_x, foot_trail_y)
            foot_trace.set_alpha(trail_alpha * 0.7)
            foot_trace_glow.set_data(foot_trail_x, foot_trail_y)
            foot_trace_glow.set_alpha(trail_alpha * 0.1)

            # Referencia
            ref_foot_line.set_data(ref_x, ref_y)
            ref_foot_line.set_alpha(trail_alpha * 0.4)

    # ── Speed indicator ──────────────────────────────
    if frame >= MOTION_START:
        if frame < FULL_SPEED:
            spd = 5.0 + 25.0 * ease((frame - MOTION_START) / (FULL_SPEED - MOTION_START))
        else:
            spd = 30.0
        speed_text.set_text(f'ω = {spd:.0f}°/s  |  θ = {angle:.0f}°')
        speed_text.set_alpha(0.6)

    # ── Fade out ─────────────────────────────────────
    if frame >= FADE_OUT:
        fade = 1.0 - ease(min(1.0, (frame - FADE_OUT) / (FPS * 3)))
        for line in bar_lines:
            line.set_alpha(line.get_alpha() * fade)
        for lbl in bar_label_texts:
            lbl.set_alpha(lbl.get_alpha() * fade)
        for artist in [foot_trace, foot_trace_glow, ref_foot_line, 
                        status_text, speed_text]:
            artist.set_alpha(artist.get_alpha() * fade)
        node_scatter.set_alpha(node_scatter.get_alpha() * fade)

    return []

print(f"Rendering Act 4: Assembly & Motion ({N_FRAMES} frames @ {FPS}fps = {DURATION_S}s)")
print(f"Output: {OUT_FILE}")

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
anim.save(OUT_FILE, writer=writer)
print(f"✓ Act 4 saved: {OUT_FILE}")
plt.close()
