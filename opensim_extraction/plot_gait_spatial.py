#!/usr/bin/env python3
"""
plot_gait_spatial.py — Trayectorias espaciales XY de cadera, rodilla, tobillo y pie.
Usa cinemática directa desde los ángulos articulares del Gait2392 de OpenSim.

Las trayectorias se grafican en el mismo estilo que el Theo Jansen
para permitir comparación directa.

Ejecutar: source ~/pytorch-rocm/bin/activate && python3 plot_gait_spatial.py
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKDIR = os.path.expanduser("~/opensim_gait_data")
CSV_PATH = os.path.join(WORKDIR, "gait_coordinates.csv")

# === Parámetros antropométricos (mm) — modelo Gait2392 ===
# Longitudes de segmentos típicas (adulto ~1.8m)
THIGH_LEN = 420.0    # fémur (mm)
SHANK_LEN = 430.0    # tibia (mm)
FOOT_LEN = 150.0     # pie hasta punta (mm)
HEEL_LEN = 50.0      # talón hacia atrás (mm)

# Velocidad de caminata horizontal para reconstruir posición X
# del Gait2392: ~1.3 m/s → pelvis_tx en los datos
HIP_HEIGHT = 920.0   # altura de la cadera desde el suelo (mm)

# === Leer datos ===
with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

time_arr = np.array([float(r['time']) for r in rows])

# Extraer ángulos (grados → radianes) — pierna derecha
hip_flex = np.array([math.radians(float(r['hip_flexion_r'])) for r in rows])
knee_ang = np.array([math.radians(float(r['knee_angle_r'])) for r in rows])
ankle_ang = np.array([math.radians(float(r['ankle_angle_r'])) for r in rows])

# Posición horizontal de la pelvis (viene directamente de OpenSim, en meters)
pelvis_tx = np.array([float(r['pelvis_tx']) for r in rows]) * 1000  # m → mm
pelvis_ty = np.array([float(r['pelvis_ty']) for r in rows]) * 1000  # m → mm

# === Cinemática directa 2D (plano sagital) ===
# Convención: X = adelante, Y = arriba
# hip_flexion > 0 → pierna adelante
# knee_angle < 0 → flexión rodilla (negativo en OpenSim Gait2392)
# ankle_angle > 0 → dorsiflexión

n = len(rows)
hip_x = pelvis_tx.copy()
hip_y = pelvis_ty.copy()

# Muslo: desde cadera, hacia abajo y adelante según hip_flexion
# Ángulo absoluto del muslo: -90° (vertical abajo) + hip_flexion
thigh_abs = -np.pi/2 + hip_flex
knee_x = hip_x + THIGH_LEN * np.cos(thigh_abs)
knee_y = hip_y + THIGH_LEN * np.sin(thigh_abs)

# Pierna: desde rodilla, ángulo relativo = thigh + knee_angle
# knee_angle es negativo para flexión → gira la pierna hacia atrás
shank_abs = thigh_abs + knee_ang
ankle_x = knee_x + SHANK_LEN * np.cos(shank_abs)
ankle_y = knee_y + SHANK_LEN * np.sin(shank_abs)

# Pie: desde tobillo, ángulo relativo = shank + ankle + 90°
foot_abs = shank_abs + ankle_ang + np.pi/2
toe_x = ankle_x + FOOT_LEN * np.cos(foot_abs)
toe_y = ankle_y + FOOT_LEN * np.sin(foot_abs)
heel_x = ankle_x - HEEL_LEN * np.cos(foot_abs)
heel_y = ankle_y - HEEL_LEN * np.sin(foot_abs)

# === Normalizar: quitar desplazamiento horizontal para ver patrón puro ===
# Centrar sobre un solo ciclo de paso
mid_x = (hip_x.min() + hip_x.max()) / 2

# Trayectoria relativa a la cadera (para ver el patrón sin traslación)
knee_rx = knee_x - hip_x
knee_ry = knee_y - hip_y
ankle_rx = ankle_x - hip_x
ankle_ry = ankle_y - hip_y
toe_rx = toe_x - hip_x
toe_ry = toe_y - hip_y

# === Gráfica 1: Trayectorias absolutas en espacio ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Trayectorias Espaciales de la Marcha — Gait2392 (Pierna Derecha)',
             color='white', fontsize=16, fontweight='bold', y=0.98)

for ax in axes.flat:
    ax.set_facecolor('#16213e')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, color='#444')
    ax.tick_params(colors='#888')
    for sp in ax.spines.values():
        sp.set_color('#333')

# Panel 1: Esqueleto completo en varias poses
ax = axes[0, 0]
ax.set_title('Secuencia de poses (cada 10 frames)', color='#e0e0e0', fontsize=11)
for i in range(0, n, 10):
    alpha = 0.15 + 0.6 * (i / n)
    ax.plot([hip_x[i], knee_x[i]], [hip_y[i], knee_y[i]], '-',
            color='#e74c3c', alpha=alpha, linewidth=2)
    ax.plot([knee_x[i], ankle_x[i]], [ankle_y[i], ankle_y[i]], '-',
            color='#2ecc71', alpha=alpha, linewidth=2)
    ax.plot([knee_x[i], ankle_x[i]], [knee_y[i], ankle_y[i]], '-',
            color='#2ecc71', alpha=alpha, linewidth=2)
    ax.plot([ankle_x[i], toe_x[i]], [ankle_y[i], toe_y[i]], '-',
            color='#3498db', alpha=alpha, linewidth=2)
ax.plot(hip_x, hip_y, '.', color='#e74c3c', markersize=1, label='Cadera')
ax.axhline(y=0, color='#00ff88', alpha=0.3, linestyle='--', linewidth=1)
ax.set_xlabel('X (mm)', color='#aaa')
ax.set_ylabel('Y (mm)', color='#aaa')
ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')

# Panel 2: Trayectoria de la rodilla (relativa a cadera)
ax = axes[0, 1]
ax.set_title('Trayectoria de RODILLA (relativa a cadera)', color='#e0e0e0', fontsize=11)
sc = ax.scatter(knee_rx, knee_ry, c=np.linspace(0, 100, n), cmap='viridis',
                s=8, alpha=0.8, zorder=3)
ax.plot(knee_rx, knee_ry, '-', color='#2ecc71', alpha=0.3, linewidth=1)
ax.plot(0, 0, 'o', color='#e74c3c', markersize=10, label='Cadera (origen)')
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Ciclo de marcha (%)', color='#aaa', fontsize=8)
cbar.ax.tick_params(labelcolor='#888')
ax.set_xlabel('ΔX (mm)', color='#aaa')
ax.set_ylabel('ΔY (mm)', color='#aaa')
ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')

# Panel 3: Trayectoria del tobillo (relativa a cadera)
ax = axes[1, 0]
ax.set_title('Trayectoria de TOBILLO (relativa a cadera)', color='#e0e0e0', fontsize=11)
sc = ax.scatter(ankle_rx, ankle_ry, c=np.linspace(0, 100, n), cmap='plasma',
                s=8, alpha=0.8, zorder=3)
ax.plot(ankle_rx, ankle_ry, '-', color='#3498db', alpha=0.3, linewidth=1)
ax.plot(0, 0, 'o', color='#e74c3c', markersize=10, label='Cadera (origen)')
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Ciclo de marcha (%)', color='#aaa', fontsize=8)
cbar.ax.tick_params(labelcolor='#888')
ax.set_xlabel('ΔX (mm)', color='#aaa')
ax.set_ylabel('ΔY (mm)', color='#aaa')
ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')

# Panel 4: Trayectoria de la PUNTA DEL PIE (relativa a cadera) — comparación con Theo Jansen
ax = axes[1, 1]
ax.set_title('Trayectoria PUNTA DEL PIE (relativa a cadera) — comparar con Jansen',
             color='#e0e0e0', fontsize=11)
sc = ax.scatter(toe_rx, toe_ry, c=np.linspace(0, 100, n), cmap='coolwarm',
                s=10, alpha=0.9, zorder=3)
ax.plot(toe_rx, toe_ry, '-', color='#9b59b6', alpha=0.4, linewidth=1)
ax.plot(0, 0, 'o', color='#e74c3c', markersize=10, label='Cadera (origen)')

# Marcar fase de apoyo (suelo) vs aérea
ground_thresh = toe_ry.min() + 30  # mm
ground_mask = toe_ry < ground_thresh
ax.scatter(toe_rx[ground_mask], toe_ry[ground_mask], s=20, color='#00ff88',
           alpha=0.5, zorder=4, label='Fase apoyo')

cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Ciclo de marcha (%)', color='#aaa', fontsize=8)
cbar.ax.tick_params(labelcolor='#888')
ax.set_xlabel('ΔX (mm)', color='#aaa')
ax.set_ylabel('ΔY (mm)', color='#aaa')
ax.legend(fontsize=8, facecolor='#1a1a2e', edgecolor='#333', labelcolor='#ccc')

plt.tight_layout(rect=[0, 0, 1, 0.95])
path1 = os.path.join(WORKDIR, "gait_spatial_trajectories.png")
fig.savefig(path1, dpi=150, facecolor=fig.get_facecolor())
print(f"[OK] {path1}")

# === Gráfica 2: Solo trayectoria del pie (formato limpio para comparar con Jansen) ===
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('#1a1a2e')
ax2.set_facecolor('#16213e')
ax2.set_aspect('equal')

ax2.set_title('Trayectoria del Pie Humano vs Patrón de Theo Jansen',
              color='white', fontsize=14, fontweight='bold')

# Trayectoria del pie humano
ax2.plot(toe_rx, toe_ry, '-', color='#3498db', linewidth=2.5, alpha=0.9,
         label='Punta del pie (humano)')
ax2.scatter(toe_rx[ground_mask], toe_ry[ground_mask], s=15, color='#00ff88',
            alpha=0.7, zorder=4, label='Fase de apoyo')
ax2.scatter(toe_rx[~ground_mask], toe_ry[~ground_mask], s=10, color='#e74c3c',
            alpha=0.5, zorder=3, label='Fase aérea')

# Marcar dirección
for i in range(0, n, 20):
    dx = toe_rx[(i+1) % n] - toe_rx[i]
    dy = toe_ry[(i+1) % n] - toe_ry[i]
    ax2.annotate('', xy=(toe_rx[i]+dx*3, toe_ry[i]+dy*3),
                xytext=(toe_rx[i], toe_ry[i]),
                arrowprops=dict(arrowstyle='->', color='#ff9f43', lw=1.5))

# Dimensiones
toe_w = toe_rx.max() - toe_rx.min()
toe_h = toe_ry.max() - toe_ry.min()
ax2.text(0.02, 0.98,
         f'Ancho: {toe_w:.0f}mm\nAlto: {toe_h:.0f}mm\n'
         f'Ratio W/H: {toe_w/toe_h:.2f}',
         transform=ax2.transAxes, color='#aaa', fontsize=10,
         va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#444'))

ax2.plot(0, 0, 'o', color='#e74c3c', markersize=12, zorder=5, label='Cadera')
ax2.axhline(y=toe_ry.min(), color='#00ff88', alpha=0.3, linestyle='--')
ax2.text(toe_rx.max() + 10, toe_ry.min() + 5, 'suelo', color='#00ff88',
         fontsize=9, alpha=0.5)

ax2.set_xlabel('ΔX relativo a cadera (mm)', color='#ccc', fontsize=11)
ax2.set_ylabel('ΔY relativo a cadera (mm)', color='#ccc', fontsize=11)
ax2.grid(True, alpha=0.15, color='#444')
ax2.tick_params(colors='#888')
for sp in ax2.spines.values():
    sp.set_color('#333')
ax2.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e',
           edgecolor='#333', labelcolor='#ccc')

plt.tight_layout()
path2 = os.path.join(WORKDIR, "gait_foot_trajectory.png")
fig2.savefig(path2, dpi=150, facecolor=fig2.get_facecolor())
print(f"[OK] {path2}")

# === Guardar trayectorias como CSV ===
traj_csv = os.path.join(WORKDIR, "spatial_trajectories.csv")
with open(traj_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['time', 'gait_pct',
                'hip_x', 'hip_y', 'knee_x', 'knee_y',
                'ankle_x', 'ankle_y', 'toe_x', 'toe_y',
                'knee_rx', 'knee_ry', 'ankle_rx', 'ankle_ry',
                'toe_rx', 'toe_ry'])
    gait_pct = (time_arr - time_arr[0]) / (time_arr[-1] - time_arr[0]) * 100
    for i in range(n):
        w.writerow([f"{time_arr[i]:.4f}", f"{gait_pct[i]:.2f}",
                    f"{hip_x[i]:.2f}", f"{hip_y[i]:.2f}",
                    f"{knee_x[i]:.2f}", f"{knee_y[i]:.2f}",
                    f"{ankle_x[i]:.2f}", f"{ankle_y[i]:.2f}",
                    f"{toe_x[i]:.2f}", f"{toe_y[i]:.2f}",
                    f"{knee_rx[i]:.2f}", f"{knee_ry[i]:.2f}",
                    f"{ankle_rx[i]:.2f}", f"{ankle_ry[i]:.2f}",
                    f"{toe_rx[i]:.2f}", f"{toe_ry[i]:.2f}"])
print(f"[OK] {traj_csv}")

# Resumen
print(f"\n{'='*60}")
print("  Dimensiones de las trayectorias (relativas a cadera)")
print(f"{'='*60}")
for name, rx, ry in [('Rodilla', knee_rx, knee_ry),
                      ('Tobillo', ankle_rx, ankle_ry),
                      ('Punta pie', toe_rx, toe_ry)]:
    w = rx.max() - rx.min()
    h = ry.max() - ry.min()
    print(f"  {name:<12s}  ancho={w:6.1f}mm  alto={h:6.1f}mm  ratio={w/h:.2f}")

print(f"\n  Theo Jansen pie: ancho=67.9mm  alto=22.5mm  ratio=3.02")
print(f"  → La trayectoria del pie humano es la META para el GAD")

# Abrir
import subprocess
subprocess.Popen(['xdg-open', path2])
print(f"\n[INFO] Abriendo gráficas...")
