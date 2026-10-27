#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  ACT 3 — Evolutionary Optimization (GAD NSGA-III)
═══════════════════════════════════════════════════════════════════════
  Usa el genoma REAL evolucionado (best_genome.json) con general_solver.
  Simula la evolución perturbando las longitudes de barras y convergiendo
  hacia el best genome final.

  - Panel izq: mecanismo animado (topology fija, longitudes evolucionando)
  - Panel sup-der: curvas de fitness reales (evolution_history.json)
  - Panel inf-der: trayectoria del pie actual vs target OpenSim

  Salida: act3_evolution.mp4  (~60s @ 30fps)
"""
import os, sys, json, math, copy
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
ref = load_reference_trajectories()
ref_foot = ref['foot']   # (360, 2) mm, relativo a cadera
ref_knee = ref['knee']

# Evolution history (500 generaciones reales)
with open(os.path.join(RESULT_DIR, 'evolution_history.json')) as f:
    history = json.load(f)

# Best genome final
with open(os.path.join(RESULT_DIR, 'best_genome.json')) as f:
    best_data = json.load(f)
g_best = LinkageGenome.from_dict(best_data)

# Longitudes finales (las que queremos alcanzar)
best_lengths = np.array([e['length'] for e in g_best.edges])

# Longitudes iniciales "malas" (perturbadas ×2 random)
rng = np.random.RandomState(42)
perturb_factors = 1.0 + rng.randn(len(best_lengths)) * 0.6
perturb_factors = np.clip(perturb_factors, 0.4, 2.5)
early_lengths = best_lengths * perturb_factors

# Función para crear edges con longitudes interpoladas
def make_edges_at_progress(progress):
    """Interpola longitudes: early → best según progress [0,1]."""
    t = progress * progress * (3 - 2 * progress)  # ease in-out
    lengths = early_lengths + (best_lengths - early_lengths) * t
    edges = copy.deepcopy(g_best.edges)
    for i, e in enumerate(edges):
        e['length'] = lengths[i]
    return edges

# Pre-calcular trayectoria final para la referencia en panel izq
trajs_best = compute_full_trajectory(g_best.nodes, g_best.edges)
foot_id = g_best.foot_id or 7
best_foot_traj = trajs_best[foot_id]

# Offset para alinear con ref (como animate_evolved.py)
mech_flipped = np.column_stack([-best_foot_traj[:, 0], best_foot_traj[:, 1]])
offset = ref_foot.mean(axis=0) - mech_flipped.mean(axis=0)
OX, OY = offset[0], offset[1]

# ── Parámetros de video ──────────────────────────────────────────────
FPS = 30
DURATION_S = 60
N_FRAMES = FPS * DURATION_S
OUT_DIR = os.path.dirname(__file__)
OUT_FILE = os.path.join(OUT_DIR, 'act3_evolution.mp4')

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
ORANGE = '#ff9f43'

# ── Fases ────────────────────────────────────────────────────────────
TITLE_END    = int(3.0 * FPS)
EVOL_START   = int(4.0 * FPS)
EVOL_END     = int(50.0 * FPS)
FINAL_HOLD   = int(52.0 * FPS)
FADE_OUT     = int(56.0 * FPS)

# ── Figura ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(BG)

ax_mech = fig.add_axes([0.03, 0.05, 0.45, 0.85])
ax_fit  = fig.add_axes([0.55, 0.55, 0.42, 0.37])
ax_traj = fig.add_axes([0.55, 0.05, 0.42, 0.42])

for ax in [ax_mech, ax_fit, ax_traj]:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color('#2a2a4a')

# ── Panel izq: mecanismo ────────────────────────────────────────────
ax_mech.set_aspect('equal')
ax_mech.set_xticks([]); ax_mech.set_yticks([])
ax_mech.set_title('Evolved Mechanism', color=WHITE, fontsize=14, fontweight='bold', pad=10)

# Precalcular bbox del mecanismo final para fijar los límites
all_final_pts = []
pos0 = solve_linkage(g_best.nodes, g_best.edges, 0)
for nid, p in pos0.items():
    all_final_pts.append((-p[0] + OX, p[1] + OY))
all_final_pts = np.array(all_final_pts)
cx_m = all_final_pts[:, 0].mean()
cy_m = all_final_pts[:, 1].mean()
rng_m = max(all_final_pts[:,0].max()-all_final_pts[:,0].min(),
            all_final_pts[:,1].max()-all_final_pts[:,1].min()) * 1.3
ax_mech.set_xlim(cx_m - rng_m, cx_m + rng_m)
ax_mech.set_ylim(cy_m - rng_m, cy_m + rng_m)

# Barras del mecanismo
mech_lines = []
for i in range(len(g_best.edges)):
    line, = ax_mech.plot([], [], '-', color=COLORS[i % len(COLORS)],
                          linewidth=4, solid_capstyle='round', alpha=0, zorder=5)
    mech_lines.append(line)

mech_joints = ax_mech.scatter([], [], s=80, c=[ORANGE], zorder=10, alpha=0,
                               edgecolors='white', linewidth=1)

mech_foot_trail, = ax_mech.plot([], [], '-', color=RED, linewidth=1.5, alpha=0, zorder=3)
mech_foot_glow, = ax_mech.plot([], [], '-', color=RED, linewidth=6, alpha=0, zorder=2)

gen_text = ax_mech.text(0.02, 0.02, '', transform=ax_mech.transAxes, fontsize=18,
                         color=CYAN, fontfamily='monospace', fontweight='bold', alpha=0)

# ── Panel sup-der: fitness ───────────────────────────────────────────
ax_fit.set_title('Fitness Convergence', color=WHITE, fontsize=12, fontweight='bold')
ax_fit.set_xlabel('Generation', color=GRAY, fontsize=9)
ax_fit.set_ylabel('Best Fitness (↓)', color=GRAY, fontsize=9)
ax_fit.tick_params(colors=GRAY, labelsize=8)
ax_fit.grid(True, color='#1a1a3a', linewidth=0.3)

gens = [h['gen'] for h in history]
fitness_vals = [h['best_fitness'] for h in history]
foot_errs = [h['best_foot'] for h in history]
knee_errs = [h['best_knee'] for h in history]

ax_fit.set_xlim(0, max(gens) + 10)
ax_fit.set_ylim(0, max(fitness_vals) * 1.1)

fit_line, = ax_fit.plot([], [], '-', color=CYAN, linewidth=2, alpha=0, label='Total fitness')
foot_line_f, = ax_fit.plot([], [], '-', color=RED, linewidth=1.5, alpha=0, label='Foot error')
knee_line_f, = ax_fit.plot([], [], '-', color=GREEN, linewidth=1.5, alpha=0, label='Knee error')
fit_dot = ax_fit.scatter([], [], s=60, c=[CYAN], zorder=10, alpha=0)
ax_fit.legend(fontsize=8, facecolor=BG, edgecolor='#333355',
              labelcolor='white', loc='upper right')

# ── Panel inf-der: trayectoria ──────────────────────────────────────
ax_traj.set_title('Foot Trajectory vs Target', color=WHITE, fontsize=12, fontweight='bold')
ax_traj.set_xlabel('X (mm)', color=GRAY, fontsize=9)
ax_traj.set_ylabel('Y (mm)', color=GRAY, fontsize=9)
ax_traj.tick_params(colors=GRAY, labelsize=8)
ax_traj.grid(True, color='#1a1a3a', linewidth=0.3)

# Reference (fixed) — flipped X + offset to match
ref_plot, = ax_traj.plot(ref_foot[:, 0], ref_foot[:, 1], '--', color=CYAN,
                          linewidth=2, alpha=0, label='Target (OpenSim)')
curr_plot, = ax_traj.plot([], [], '-', color=RED, linewidth=2, alpha=0, label='Current best')
ax_traj.legend(fontsize=8, facecolor=BG, edgecolor='#333355',
               labelcolor='white', loc='upper right')

pad = 100
ax_traj.set_xlim(ref_foot[:,0].min()-pad, ref_foot[:,0].max()+pad)
ax_traj.set_ylim(ref_foot[:,1].min()-pad, ref_foot[:,1].max()+pad)

# Título principal
main_title = fig.text(0.5, 0.97, '', fontsize=28, color=WHITE,
                       ha='center', va='top', fontweight='bold', alpha=0)

# Métricas
metrics_text = fig.text(0.55, 0.47, '', fontsize=10, color='#9999bb',
                         fontfamily='monospace', alpha=0)

# ── Animación ────────────────────────────────────────────────────────
def ease(t):
    return t * t * (3 - 2*t)

def animate(frame):
    # ── Título ───────────────────────────────────────
    if frame < TITLE_END:
        a_t = ease(min(1.0, frame / (FPS * 1.5)))
        main_title.set_text('Evolutionary Optimization')
        main_title.set_alpha(a_t)
    elif frame < TITLE_END + FPS:
        main_title.set_alpha(1.0 - ease((frame - TITLE_END) / FPS))
    else:
        main_title.set_alpha(0)

    # ── Evolución progresiva ─────────────────────────
    if frame >= EVOL_START:
        evol_progress = min(1.0, (frame - EVOL_START) / (EVOL_END - EVOL_START))
        gen_idx = int(evol_progress * (len(history) - 1))
        current_gen = history[gen_idx]
        gen_num = current_gen['gen']

        # Crear edges con longitudes interpoladas
        edges_now = make_edges_at_progress(evol_progress)
        nodes_now = g_best.nodes  # topología fija

        # Crank angle animado
        walk_phase = ((frame - EVOL_START) / FPS * 2.0) % 1.0
        crank_angle = walk_phase * 360.0

        # Resolver posiciones
        positions = solve_linkage(nodes_now, edges_now, crank_angle)
        alpha_e = ease(min(1.0, (frame - EVOL_START) / (FPS * 2)))

        if positions is not None:
            # Dibujar barras
            for i, e in enumerate(edges_now):
                n1, n2 = e['n1'], e['n2']
                if n1 in positions and n2 in positions:
                    p1 = (-positions[n1][0] + OX, positions[n1][1] + OY)
                    p2 = (-positions[n2][0] + OX, positions[n2][1] + OY)
                    mech_lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    mech_lines[i].set_alpha(alpha_e)

            # Joints
            valid_pts = []
            for n in nodes_now:
                nid = n['id']
                if nid in positions:
                    valid_pts.append((-positions[nid][0] + OX, positions[nid][1] + OY))
            if valid_pts:
                mech_joints.set_offsets(valid_pts)
                mech_joints.set_alpha(alpha_e)

            # Foot trajectory para este set de longitudes
            trajs_now = compute_full_trajectory(nodes_now, edges_now)
            if trajs_now is not None and foot_id in trajs_now:
                ft = trajs_now[foot_id]
                ft_disp_x = -ft[:, 0] + OX
                ft_disp_y = ft[:, 1] + OY

                mech_foot_trail.set_data(ft_disp_x, ft_disp_y)
                mech_foot_trail.set_alpha(alpha_e * 0.5)
                mech_foot_glow.set_data(ft_disp_x, ft_disp_y)
                mech_foot_glow.set_alpha(alpha_e * 0.08)

                # Panel trayectoria: centrar y comparar con ref
                # Flip + offset igual que animate_evolved.py
                ft_flipped = np.column_stack([-ft[:, 0], ft[:, 1]])
                ft_offset = ref_foot.mean(axis=0) - ft_flipped.mean(axis=0)
                ft_aligned = ft_flipped + ft_offset

                curr_plot.set_data(ft_aligned[:, 0], ft_aligned[:, 1])
                curr_plot.set_alpha(alpha_e)
        else:
            # Si no ensambla, ocultar
            for line in mech_lines:
                line.set_alpha(0)

        # Gen text
        gen_text.set_text(f'Gen {gen_num:03d} / 500')
        gen_text.set_alpha(alpha_e)

        # Fitness curves
        n_show = gen_idx + 1
        fit_line.set_data(gens[:n_show], fitness_vals[:n_show])
        fit_line.set_alpha(alpha_e)
        foot_line_f.set_data(gens[:n_show], foot_errs[:n_show])
        foot_line_f.set_alpha(alpha_e * 0.7)
        knee_line_f.set_data(gens[:n_show], knee_errs[:n_show])
        knee_line_f.set_alpha(alpha_e * 0.7)

        fit_dot.set_offsets([[gens[gen_idx], fitness_vals[gen_idx]]])
        fit_dot.set_alpha(alpha_e)

        ref_plot.set_alpha(alpha_e * 0.6)

        # Metrics text
        metrics_text.set_text(
            f"Gen {gen_num:>3d}  |  Fitness: {current_gen['best_fitness']:.4f}\n"
            f"Foot err: {current_gen['best_foot']:.4f}  |  "
            f"Knee err: {current_gen['best_knee']:.4f}\n"
            f"Stride: {current_gen['best_stride']:.0f} mm  |  "
            f"Valid: {current_gen['n_valid']}"
        )
        metrics_text.set_alpha(alpha_e * 0.8)

    # ── Fade out ─────────────────────────────────────
    if frame >= FADE_OUT:
        fade = 1.0 - ease(min(1.0, (frame - FADE_OUT) / (FPS * 3)))
        for line in mech_lines:
            line.set_alpha(line.get_alpha() * fade)
        for artist in [mech_foot_trail, mech_foot_glow, fit_line, foot_line_f,
                        knee_line_f, ref_plot, curr_plot, gen_text, metrics_text]:
            artist.set_alpha(artist.get_alpha() * fade)
        mech_joints.set_alpha(mech_joints.get_alpha() * fade)
        fit_dot.set_alpha(fit_dot.get_alpha() * fade)

    return []

print(f"Rendering Act 3: Evolution ({N_FRAMES} frames @ {FPS}fps = {DURATION_S}s)")
print(f"  Using REAL genome topology from best_genome.json")
print(f"  {len(g_best.edges)} edges, foot_id={foot_id}")
print(f"  Lengths: early={early_lengths[:3].round(1)}... → best={best_lengths[:3].round(1)}...")
print(f"Output: {OUT_FILE}")

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000/FPS, blit=False)
writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
anim.save(OUT_FILE, writer=writer)
print(f"✓ Act 3 saved: {OUT_FILE}")
plt.close()
