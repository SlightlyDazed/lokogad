#!/usr/bin/env python3
"""
theo_jansen.py — Mecanismo de Theo Jansen: animación moderna (Python 3 + matplotlib)
Adaptado del código original PyGTK2 de heltonbiker (StackOverflow CC BY-SA 3.0).

El mecanismo de Jansen usa 13 longitudes de eslabón ("magic numbers") para
crear una trayectoria de pie que simula una caminata natural usando solo
un movimiento rotacional de entrada (manivela).

Puntos fijos: A (centro manivela), B (soporte del chasis)
Punto de salida: H (pie — traza la trayectoria de caminata)

Ejecutar: source ~/pytorch-rocm/bin/activate && python3 theo_jansen.py
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Renderizar sin GUI — genera GIF directamente
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

OUTPUT_DIR = os.path.expanduser("~/opensim_gait_data/theo_jansen")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================
#  Geometría del mecanismo
# =========================================================================

# Los 13 "magic numbers" de Theo Jansen (longitudes de eslabón en mm)
MAGIC_NUMBERS = [38, 7.8, 15, 50, 41.5, 39.3, 61.9, 55.8, 40.1, 39.4, 36.7, 65.7, 49]

# Descripción de cada número:
# [0]  = distancia horizontal A→B (chasis)
# [1]  = distancia vertical A→B (chasis)
# [2]  = longitud de la manivela AC
# [3]  = eslabón CD
# [4]  = eslabón BD
# [5]  = eslabón BE
# [6]  = eslabón CE
# [7]  = eslabón DF
# [8]  = eslabón BF
# [9]  = eslabón FG
# [10] = eslabón EG
# [11] = eslabón GH
# [12] = eslabón EH


def linkage(p1, l1, p2, l2):
    """
    Calcula la intersección de dos círculos (p1,l1) y (p2,l2).
    Retorna el punto que está al lado correcto (CCW test).
    Esta es la función clave que resuelve la cinemática del mecanismo.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = math.sqrt(dx**2 + dy**2)

    if d > l1 + l2 or d < abs(l1 - l2) or d == 0:
        return None  # No hay solución

    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h = math.sqrt(max(0, l1**2 - a**2))

    mx = p1[0] + dx * a / d
    my = p1[1] + dy * a / d

    rx = -dy * h / d
    ry = dx * h / d

    r1 = (mx + rx, my + ry)
    r2 = (mx - rx, my - ry)

    # CCW test: elegir el punto al lado correcto de la línea p1→p2
    cross = (p2[0] - p1[0]) * (r1[1] - p1[1]) - (p2[1] - p1[1]) * (r1[0] - p1[0])
    if cross < 0:
        return r1
    else:
        return r2


def assemble(angle_deg, mutations=None):
    """
    Ensambla el mecanismo de Jansen para un ángulo de manivela dado.
    Retorna un diccionario con todos los puntos y eslabones.
    """
    mn = list(MAGIC_NUMBERS)
    if mutations:
        mn = [mn[i] + mutations[i] for i in range(13)]

    # Puntos fijos
    A = (0.0, 0.0)
    B = (-mn[0], -mn[1])

    # Manivela: C rota alrededor de A
    angle_rad = math.radians(angle_deg)
    C = (A[0] + mn[2] * math.cos(angle_rad),
         A[1] + mn[2] * math.sin(angle_rad))

    # Triángulo superior: D = intersección de círculos (C, mn[3]) y (B, mn[4])
    D = linkage(C, mn[3], B, mn[4])
    # Triángulo inferior: E = intersección de círculos (B, mn[5]) y (C, mn[6])
    E = linkage(B, mn[5], C, mn[6])
    # F = intersección de círculos (D, mn[7]) y (B, mn[8])
    F = linkage(D, mn[7], B, mn[8])
    # G = intersección de círculos (F, mn[9]) y (E, mn[10])
    G = linkage(F, mn[9], E, mn[10])
    # H (pie) = intersección de círculos (G, mn[11]) y (E, mn[12])
    H = linkage(G, mn[11], E, mn[12])

    points = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H}

    # Eslabones (conexiones entre puntos)
    links = [
        ('A', 'C'),   # manivela
        ('C', 'D'),   # eslabón CD
        ('B', 'D'),   # eslabón BD
        ('B', 'E'),   # eslabón BE
        ('C', 'E'),   # eslabón CE
        ('D', 'F'),   # eslabón DF
        ('B', 'F'),   # eslabón BF
        ('F', 'G'),   # eslabón FG
        ('E', 'G'),   # eslabón EG
        ('G', 'H'),   # eslabón GH
        ('E', 'H'),   # eslabón EH
    ]

    return points, links


def compute_foot_path(n_steps=360, mutations=None):
    """Calcula la trayectoria completa del pie H."""
    path = []
    for angle in range(n_steps):
        pts, _ = assemble(angle, mutations)
        if pts['H'] is not None:
            path.append(pts['H'])
    return path


# =========================================================================
#  Animación con matplotlib
# =========================================================================

def create_animation(save_gif=True):
    """Crea la animación interactiva del mecanismo de Jansen."""

    # Pre-calcular trayectoria del pie
    foot_path = compute_foot_path()
    fx = [p[0] for p in foot_path]
    fy = [p[1] for p in foot_path]

    # Configuración de la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1, 1.5]})
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Mecanismo de Theo Jansen — Simulación GAD',
                 color='white', fontsize=16, fontweight='bold', y=0.97)

    # --- Panel izquierdo: mecanismo ---
    ax1.set_facecolor('#16213e')
    ax1.set_aspect('equal')
    ax1.set_title('Mecanismo (8 puntos, 11 eslabones)', color='#e0e0e0', fontsize=12)
    ax1.tick_params(colors='#666666')
    for spine in ax1.spines.values():
        spine.set_color('#333333')
    ax1.grid(True, alpha=0.15, color='#444444')

    # Líneas de los eslabones
    link_lines, = ax1.plot([], [], 'o-', color='#00d2ff', linewidth=2,
                           markersize=6, markerfacecolor='#ffffff',
                           markeredgecolor='#00d2ff', markeredgewidth=1.5)
    # Línea del pie (H) resaltada
    foot_marker, = ax1.plot([], [], 'o', color='#ff6b6b', markersize=10,
                            markeredgecolor='white', markeredgewidth=2, zorder=5)
    # Puntos fijos A y B
    fixed_markers, = ax1.plot([], [], 's', color='#ffd93d', markersize=8,
                              markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    # Trayectoria del pie (ghost)
    trail_line, = ax1.plot(fx, fy, '-', color='#ff6b6b', alpha=0.3, linewidth=1)
    # Etiquetas de los puntos
    labels = {}
    for name in 'ABCDEFGH':
        labels[name] = ax1.text(0, 0, f' {name}', fontsize=8, color='#aaaaaa',
                                fontweight='bold', ha='left', va='bottom')
    # Texto del ángulo
    angle_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                          color='#00d2ff', fontsize=11, va='top',
                          fontfamily='monospace')

    ax1.set_xlim(-90, 40)
    ax1.set_ylim(-120, 30)

    # --- Panel derecho: trayectoria del pie ---
    ax2.set_facecolor('#16213e')
    ax2.set_aspect('equal')
    ax2.set_title('Trayectoria del pie (punto H)', color='#e0e0e0', fontsize=12)
    ax2.tick_params(colors='#666666')
    for spine in ax2.spines.values():
        spine.set_color('#333333')
    ax2.grid(True, alpha=0.15, color='#444444')

    # Trayectoria completa
    ax2.plot(fx, fy, '-', color='#444444', linewidth=1, alpha=0.5)

    # Fase de apoyo (contacto con suelo) vs fase aérea
    # Detectar la parte inferior de la trayectoria (fase de apoyo)
    min_y = min(fy)
    threshold = min_y + 3  # margen de 3mm
    ground_x, ground_y = [], []
    air_x, air_y = [], []
    for i, (x, y) in enumerate(zip(fx, fy)):
        if y < threshold:
            ground_x.append(x)
            ground_y.append(y)
        else:
            air_x.append(x)
            air_y.append(y)

    ax2.plot(ground_x, ground_y, '.', color='#00ff88', markersize=3,
             alpha=0.7, label='Fase de apoyo (suelo)')
    ax2.plot(air_x, air_y, '.', color='#ff6b6b', markersize=2,
             alpha=0.5, label='Fase aérea')

    # Pie actual
    foot_trace, = ax2.plot([], [], 'o', color='#ffd93d', markersize=12,
                           markeredgecolor='white', markeredgewidth=2, zorder=5)
    # Trail del pie (últimos N frames)
    trail_n = 40
    foot_trail, = ax2.plot([], [], '-', color='#00d2ff', linewidth=2, alpha=0.6)

    ax2.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e',
               edgecolor='#333333', labelcolor='#cccccc')

    pad = 5
    ax2.set_xlim(min(fx) - pad, max(fx) + pad)
    ax2.set_ylim(min(fy) - pad, max(fy) + pad)

    # Línea del suelo
    ax2.axhline(y=min_y, color='#00ff88', alpha=0.3, linestyle='--', linewidth=1)
    ax2.text(min(fx), min_y + 1, '  nivel del suelo', color='#00ff88',
             fontsize=8, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # --- Función de animación ---
    def init():
        link_lines.set_data([], [])
        foot_marker.set_data([], [])
        fixed_markers.set_data([], [])
        foot_trace.set_data([], [])
        foot_trail.set_data([], [])
        angle_text.set_text('')
        return (link_lines, foot_marker, fixed_markers, foot_trace,
                foot_trail, angle_text)

    def animate(frame):
        angle = frame * 3  # 3° por frame → 120 frames = ciclo completo
        pts, links = assemble(angle)

        if pts['H'] is None:
            return (link_lines, foot_marker, fixed_markers, foot_trace,
                    foot_trail, angle_text)

        # Dibujar eslabones siguiendo el orden de conexión del mecanismo
        all_x, all_y = [], []
        # Recorrer el mecanismo en un orden que dibuje todos los eslabones
        draw_order = [
            ['A', 'C', 'D', 'B'],       # triángulo superior
            ['B', 'E', 'C'],             # triángulo inferior
            ['B', 'F', 'D'],             # eslabón BF-DF
            ['F', 'G', 'E'],             # eslabón FG-EG
            ['E', 'H', 'G'],             # eslabón EH-GH (pie)
        ]
        for chain in draw_order:
            for name in chain:
                p = pts[name]
                if p is not None:
                    all_x.append(p[0])
                    all_y.append(p[1])
            # Separador (NaN) para no conectar cadenas
            all_x.append(float('nan'))
            all_y.append(float('nan'))

        link_lines.set_data(all_x, all_y)

        # Pie (punto H)
        hx, hy = pts['H']
        foot_marker.set_data([hx], [hy])

        # Puntos fijos
        fixed_markers.set_data([pts['A'][0], pts['B'][0]],
                               [pts['A'][1], pts['B'][1]])

        # Etiquetas
        for name, pt in pts.items():
            if pt is not None:
                labels[name].set_position((pt[0], pt[1]))

        angle_text.set_text(f'θ = {angle % 360:3d}°')

        # Trail del pie en panel derecho
        foot_trace.set_data([hx], [hy])
        idx = (angle % 360)
        start = max(0, idx - trail_n)
        foot_trail.set_data(fx[start:idx+1], fy[start:idx+1])

        return (link_lines, foot_marker, fixed_markers, foot_trace,
                foot_trail, angle_text)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=120, interval=50, blit=False
    )

    # Guardar GIF
    if save_gif:
        gif_path = os.path.join(OUTPUT_DIR, "theo_jansen_animation.gif")
        print(f"[INFO] Guardando GIF en {gif_path}...")
        try:
            anim.save(gif_path, writer='pillow', fps=20, dpi=100)
            print(f"[OK] GIF guardado ({os.path.getsize(gif_path):,d} bytes)")
        except Exception as e:
            print(f"[WARN] No se pudo guardar GIF: {e}")

    # Abrir el GIF con el visor predeterminado del sistema
    if save_gif and os.path.exists(gif_path):
        import subprocess
        print("[INFO] Abriendo GIF...")
        subprocess.Popen(['xdg-open', gif_path])


# =========================================================================
#  Análisis del mecanismo
# =========================================================================

def analyze():
    """Imprime análisis del mecanismo."""
    foot_path = compute_foot_path()
    fx = [p[0] for p in foot_path]
    fy = [p[1] for p in foot_path]

    print("=" * 60)
    print("  Análisis del mecanismo de Theo Jansen")
    print("=" * 60)

    print(f"\n  Magic Numbers (longitudes de eslabón, mm):")
    labels = ['A→B horiz', 'A→B vert', 'Manivela AC', 'CD', 'BD',
              'BE', 'CE', 'DF', 'BF', 'FG', 'EG', 'GH', 'EH']
    for i, (val, label) in enumerate(zip(MAGIC_NUMBERS, labels)):
        print(f"    [{i:2d}] {val:6.1f} mm  — {label}")

    print(f"\n  Trayectoria del pie (punto H):")
    print(f"    Rango X: {min(fx):.1f} a {max(fx):.1f} mm (ancho: {max(fx)-min(fx):.1f} mm)")
    print(f"    Rango Y: {min(fy):.1f} a {max(fy):.1f} mm (alto: {max(fy)-min(fy):.1f} mm)")

    min_y = min(fy)
    threshold = min_y + 3
    ground_count = sum(1 for y in fy if y < threshold)
    air_count = len(fy) - ground_count
    print(f"\n  Fases de la caminata:")
    print(f"    Fase de apoyo: {ground_count}° ({ground_count/360*100:.0f}% del ciclo)")
    print(f"    Fase aérea:    {air_count}° ({air_count/360*100:.0f}% del ciclo)")

    # Velocidad del pie en fase de apoyo
    ground_fx = [(fx[i], fy[i]) for i in range(len(fx)) if fy[i] < threshold]
    if len(ground_fx) > 1:
        stride = abs(ground_fx[0][0] - ground_fx[-1][0])
        print(f"    Longitud de zancada: ~{stride:.1f} mm")

    # Ejemplo de un frame
    print(f"\n  Ejemplo — ángulo = 0°:")
    pts, links = assemble(0)
    for name, pt in pts.items():
        if pt:
            print(f"    {name}: ({pt[0]:7.2f}, {pt[1]:7.2f}) mm")

    print()


# =========================================================================
#  Main
# =========================================================================

if __name__ == "__main__":
    analyze()
    create_animation(save_gif=True)
