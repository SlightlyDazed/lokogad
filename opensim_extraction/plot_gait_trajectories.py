#!/usr/bin/env python3
"""
plot_gait_trajectories.py — Extrae y grafica trayectorias articulares
de cadera, rodilla y tobillo del archivo gait_coordinates.csv de OpenSim.

Ejecutar: source ~/pytorch-rocm/bin/activate && python3 plot_gait_trajectories.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKDIR = os.path.expanduser("~/opensim_gait_data")
CSV_PATH = os.path.join(WORKDIR, "gait_coordinates.csv")
OUT_DIR = WORKDIR

# Leer datos
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

cols = list(rows[0].keys())
time_vals = np.array([float(r['time']) for r in rows])

# Normalizar tiempo al ciclo de marcha (0-100%)
t_start, t_end = time_vals[0], time_vals[-1]
gait_pct = (time_vals - t_start) / (t_end - t_start) * 100

# Articulaciones a graficar
joints = {
    'Cadera (Hip Flexion)': {
        'right': 'hip_flexion_r', 'left': 'hip_flexion_l',
        'ylabel': 'Ángulo (°)', 'color_r': '#e74c3c', 'color_l': '#c0392b',
        'desc': 'Flexión (+) / Extensión (-)'
    },
    'Rodilla (Knee Angle)': {
        'right': 'knee_angle_r', 'left': 'knee_angle_l',
        'ylabel': 'Ángulo (°)', 'color_r': '#2ecc71', 'color_l': '#27ae60',
        'desc': 'Flexión (+)'
    },
    'Tobillo (Ankle Angle)': {
        'right': 'ankle_angle_r', 'left': 'ankle_angle_l',
        'ylabel': 'Ángulo (°)', 'color_r': '#3498db', 'color_l': '#2980b9',
        'desc': 'Dorsiflexión (+) / Plantarflexión (-)'
    },
}

# === Gráfica principal: 3 subplots ===
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Trayectorias Articulares — Gait2392 (Caminata Normal)',
             color='white', fontsize=16, fontweight='bold', y=0.98)

for idx, (title, info) in enumerate(joints.items()):
    ax = axes[idx]
    ax.set_facecolor('#16213e')

    r_data = np.array([float(r[info['right']]) for r in rows])
    l_data = np.array([float(r[info['left']]) for r in rows])

    ax.plot(gait_pct, r_data, color=info['color_r'], linewidth=2.5,
            label=f'Derecha ({info["right"]})', alpha=0.9)
    ax.plot(gait_pct, l_data, color=info['color_l'], linewidth=2.5,
            linestyle='--', label=f'Izquierda ({info["left"]})', alpha=0.8)

    # Rellenar entre curvas
    ax.fill_between(gait_pct, r_data, l_data, alpha=0.1, color=info['color_r'])

    ax.set_title(f'{title}  —  {info["desc"]}', color='#e0e0e0', fontsize=12,
                 fontweight='bold', loc='left')
    ax.set_ylabel(info['ylabel'], color='#cccccc', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e',
              edgecolor='#333', labelcolor='#ccc')
    ax.grid(True, alpha=0.15, color='#444')
    ax.axhline(y=0, color='#666', linewidth=0.5, linestyle='-')
    ax.tick_params(colors='#888')
    for spine in ax.spines.values():
        spine.set_color('#333')

    # Marcar fases de la marcha
    ax.axvspan(0, 60, alpha=0.03, color='green', label='Apoyo ~60%')
    ax.axvspan(60, 100, alpha=0.03, color='red', label='Balanceo ~40%')

    # Estadísticas
    stats = (f"R: [{r_data.min():.1f}°, {r_data.max():.1f}°] "
             f"rango={r_data.max()-r_data.min():.1f}°  |  "
             f"L: [{l_data.min():.1f}°, {l_data.max():.1f}°] "
             f"rango={l_data.max()-l_data.min():.1f}°")
    ax.text(0.01, 0.02, stats, transform=ax.transAxes, fontsize=7,
            color='#999', fontfamily='monospace', va='bottom')

axes[-1].set_xlabel('Ciclo de Marcha (%)', color='#cccccc', fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Guardar
png_path = os.path.join(OUT_DIR, "gait_trajectories.png")
fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
print(f"[OK] Gráfica guardada: {png_path}")

# === CSV limpio con solo las 3 articulaciones ===
out_csv = os.path.join(OUT_DIR, "joint_trajectories.csv")
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'gait_pct',
                     'hip_flexion_r', 'hip_flexion_l',
                     'knee_angle_r', 'knee_angle_l',
                     'ankle_angle_r', 'ankle_angle_l'])
    for i, r in enumerate(rows):
        writer.writerow([
            r['time'], f"{gait_pct[i]:.2f}",
            r['hip_flexion_r'], r['hip_flexion_l'],
            r['knee_angle_r'], r['knee_angle_l'],
            r['ankle_angle_r'], r['ankle_angle_l'],
        ])
print(f"[OK] CSV limpio: {out_csv}")

# === Resumen ===
print(f"\n{'='*60}")
print("  Resumen de trayectorias articulares")
print(f"{'='*60}")
for title, info in joints.items():
    r = np.array([float(row[info['right']]) for row in rows])
    l = np.array([float(row[info['left']]) for row in rows])
    print(f"\n  {title}:")
    print(f"    Derecha:   min={r.min():7.2f}°  max={r.max():7.2f}°  rango={r.max()-r.min():.2f}°")
    print(f"    Izquierda: min={l.min():7.2f}°  max={l.max():7.2f}°  rango={l.max()-l.min():.2f}°")

print(f"\n  Tiempo: {time_vals[0]:.3f}s → {time_vals[-1]:.3f}s ({len(rows)} frames)")
print(f"  Archivo fuente: {CSV_PATH}")

# Abrir imagen
import subprocess
subprocess.Popen(['xdg-open', png_path])
print(f"\n[INFO] Abriendo gráfica...")
