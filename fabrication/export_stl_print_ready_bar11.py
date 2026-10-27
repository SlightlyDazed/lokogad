#!/usr/bin/env python3
"""
export_stl_print_ready.py — STLs para impresión 3D del mecanismo evolucionado.

CARACTERÍSTICAS:
  · Escala automática: todo el recorrido del mecanismo cabe en 250×250 mm.
  · Barras con HUECOS en extremos (anillos, sin booleanos) para recibir pines.
  · Pines con longitud exacta al stack de capas que atraviesan.
  · Sin interferencias: PIN_R < HOLE_R siempre.

Uso:
    source ~/pytorch-rocm/bin/activate && python3 export_stl_print_ready.py
"""
import os, sys, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import RESULT_DIR

import trimesh

# ═══════════════════════════════════════════════════════════════════════
# PARÁMETROS DE IMPRESIÓN
# ═══════════════════════════════════════════════════════════════════════
TARGET_XY_MM   = 250.0   # caja de trabajo en mm (25 cm)
PRINT_TOL      = 0.4     # tolerancia holgura pin-hueco (mm)

# Proporciones relativas al mecanismo escalado
# (se calculan a partir de la barra más corta escalada)
BAR_THICK_REL  = 0.12    # espesor = 12% de la barra más corta
BAR_WIDTH_REL  = 0.30    # ancho   = 30% de la barra más corta
# Mínimos absolutos imprimibles (mm)
BAR_THICK_MIN  = 3.0
BAR_WIDTH_MIN  = 6.0
PIN_R_MIN      = 2.0
PIN_R_FRAC     = 0.35    # radio pin = 35% del radio del hueco de la barra

OUT_DIR = os.path.join(RESULT_DIR, "stl_for_print", "barras_redim_1.25x")

SECTIONS = 32   # resolución cilindros


# ═══════════════════════════════════════════════════════════════════════
# Z-layers (idéntico a animate_evolved.py)
# ═══════════════════════════════════════════════════════════════════════
def compute_z_layers(edges):
    n = len(edges)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ni = {edges[i]['n1'], edges[i]['n2']}
            nj = {edges[j]['n1'], edges[j]['n2']}
            if ni & nj:
                conflicts[i].add(j)
                conflicts[j].add(i)
    layers = [None] * n
    for i in range(n):
        used = {layers[j] for j in conflicts[i] if layers[j] is not None}
        for k in range(n + 1):
            candidate = (k + 1) // 2 * (1 if k % 2 == 0 else -1)
            if candidate not in used:
                layers[i] = candidate
                break
    return layers


# ═══════════════════════════════════════════════════════════════════════
# BARRA CON HUECOS — perfil 2D Shapely extruido en Z
#
# Geometría (vista desde arriba, en XY):
#
#    ╭─────────────────────────────────────╮
#    │  ◯   [═══════════════════════]   ◯  │   ← stadiumShape con huecos
#    ╰─────────────────────────────────────╯
#
#  El perfil es: (rectángulo ∪ círculo_n1 ∪ círculo_n2)  −  hueco_n1  −  hueco_n2
#  Todo en un único polígono Shapely → extruido BAR_THICK mm en Z.
#  UNA SOLA PIEZA SÓLIDA sin costuras ni uniones débiles.
#
#  eje X: de n1 (x=0) a n2 (x=L)
#  eje Z: espesor BAR_THICK, centrado en 0
# ═══════════════════════════════════════════════════════════════════════
def make_bar(L_mm, bar_width, bar_thick, hole_r):
    """
    L_mm     : longitud entre centros de pivotes (mm)
    bar_width: ancho total = diámetro de los extremos circulares (mm)
    bar_thick: espesor Z (mm)
    hole_r   : radio del hueco en cada extremo para el pin (mm)
    """
    from shapely.geometry import Point
    from shapely.geometry import box as shbox

    hw = bar_width / 2.0   # radio del semicírculo exterior

    # 1) Perfil exterior: estadio (rectángulo + dos círculos en extremos)
    outer_rect = shbox(0.0, -hw, L_mm, hw)
    outer_cap1 = Point(0.0,   0.0).buffer(hw, resolution=SECTIONS)
    outer_cap2 = Point(L_mm,  0.0).buffer(hw, resolution=SECTIONS)
    outer = outer_rect.union(outer_cap1).union(outer_cap2)

    # 2) Huecos para el eje del motor (Perfil Doble-D)
    import shapely.affinity
    # Radio es hole_r (3.0mm = diametro 6.0mm)
    # Grosor total del motor = 4.5mm (distancia entre lados planos)
    # Por lo tanto, cortamos el circulo en Y = +/- 2.25mm
    motor_circle = Point(0.0, 0.0).buffer(hole_r, resolution=SECTIONS)
    motor_rect = shbox(-hole_r, -2.25, hole_r, 2.25)
    motor_profile = motor_circle.intersection(motor_rect)

    hole1 = motor_profile
    # El hueco 2 mantiene el tamaño original (radio 2.4mm -> diam 4.8mm)
    # para que encaje el pin estándar de 4.0mm.
    hole2 = Point(L_mm, 0.0).buffer(2.4, resolution=SECTIONS)

    # 3) Perfil final: estadio menos los dos huecos
    profile = outer.difference(hole1).difference(hole2)

    # 4) Extruir el perfil 2D a lo largo de Z (BAR_THICK mm)
    mesh = trimesh.creation.extrude_polygon(profile, bar_thick)

    # Centrar en Z=0  (extrude_polygon va de 0 a bar_thick)
    mesh.apply_translation([0.0, 0.0, -bar_thick / 2.0])

    # 5) Boquilla N1 — pared DOBLE, atraviesa grosor completo de la barra
    # Superficie contacto = pared lateral a lo largo de bar_thick + 10mm externos
    boss_r1 = hole_r + 2.0 * (hw - hole_r)  # radio exterior boquilla N1
    boss_outer = Point(0.0, 0.0).buffer(boss_r1, resolution=SECTIONS)
    boss_profile1 = boss_outer.difference(hole1)
    boss_height1  = bar_thick + 10.0         # atraviesa barra + 10mm sobresale
    boss_mesh1 = trimesh.creation.extrude_polygon(boss_profile1, boss_height1)
    # Arranca desde la cara INFERIOR (-bar_thick/2) y sale por la cara superior
    boss_mesh1.apply_translation([0.0, 0.0, -bar_thick / 2.0])
    
    # 6) Boquilla N2 — pared DOBLE, atraviesa grosor completo de la barra
    # Superficie contacto = pared lateral a lo largo de bar_thick + 5mm externos
    boss_r2 = 2.4 + 2.0 * (hw - 2.4)        # radio exterior boquilla N2 (pared x2)
    boss2_outer = Point(L_mm, 0.0).buffer(boss_r2, resolution=SECTIONS)
    boss_profile2 = boss2_outer.difference(hole2)
    boss_height2  = bar_thick + 5.0          # atraviesa barra + 5mm sobresale abajo
    boss_mesh2 = trimesh.creation.extrude_polygon(boss_profile2, boss_height2)
    # Arranca 5mm por debajo de la cara inferior y sube hasta la cara superior
    boss_mesh2.apply_translation([0.0, 0.0, -bar_thick / 2.0 - 5.0])
    
    # Unir la barra principal y las dos boquillas
    mesh = trimesh.util.concatenate([mesh, boss_mesh1, boss_mesh2])

    return mesh


# ═══════════════════════════════════════════════════════════════════════
# PIN — cilindro sólido
# ═══════════════════════════════════════════════════════════════════════
def make_pin(height_mm, pin_r):
    pin = trimesh.creation.cylinder(radius=pin_r, height=height_mm,
                                     sections=SECTIONS)
    return pin


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    # ── Cargar genoma ────────────────────────────────────────────
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    if not os.path.exists(genome_path):
        print(f"[ERROR] No encontrado: {genome_path}")
        sys.exit(1)

    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)

    # ── Calcular bounding box del recorrido completo ─────────────
    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        print("[ERROR] El mecanismo no ensambla")
        sys.exit(1)

    all_x = np.concatenate([t[:, 0] for t in trajs.values()])
    all_y = np.concatenate([t[:, 1] for t in trajs.values()])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    mech_size = max(x_range, y_range)

    SCALE = TARGET_XY_MM / mech_size   # factor de escala mm/mm

    print(f"\n{'='*60}")
    print(f"  export_stl_print_ready.py")
    print(f"{'='*60}")
    print(f"  Mecanismo original:  {x_range:.0f} × {y_range:.0f} mm")
    print(f"  Factor de escala:    {SCALE:.4f}  ({1/SCALE:.1f}x reducción)")
    print(f"  Mecanismo escalado:  {x_range*SCALE:.0f} × {y_range*SCALE:.0f} mm")

    # ── Calcular dimensiones de sección transversal ──────────────
    lengths_scaled = [e['length'] * SCALE for e in g.edges]
    min_L = min(lengths_scaled)

    BAR_WIDTH = max(BAR_WIDTH_MIN, min_L * BAR_WIDTH_REL)
    BAR_THICK = max(BAR_THICK_MIN, min_L * BAR_THICK_REL)
    hw        = BAR_WIDTH / 2.0
    HOLE_R    = hw * 0.65          # hueco = 65% del radio exterior
    HOLE_R    = max(HOLE_R, PIN_R_MIN + PRINT_TOL)
    PIN_R     = max(PIN_R_MIN, HOLE_R - PRINT_TOL)

    LAYER_PITCH = BAR_THICK * 2.2  # separación entre capas ≥ 2×espesor

    print(f"\n  SECCIÓN TRANSVERSAL (constante en todas las barras):")
    print(f"  · BAR_WIDTH  = {BAR_WIDTH:.2f} mm")
    print(f"  · BAR_THICK  = {BAR_THICK:.2f} mm")
    print(f"  · HOLE_R     = {HOLE_R:.2f} mm  (hueco en extremos)")
    print(f"  · PIN_R      = {PIN_R:.2f} mm  (radio del pin)")
    print(f"  · Holgura    = {HOLE_R - PIN_R:.2f} mm  ← pin entra sin interferencia ✓")
    print(f"  · LAYER_PITCH= {LAYER_PITCH:.2f} mm")

    # ── Z-layers (preferir versión corregida geométricamente) ────
    fixed_path = os.path.join(RESULT_DIR, "z_layers_fixed.json")
    if os.path.exists(fixed_path):
        with open(fixed_path) as f:
            fixed = json.load(f)
        z_layers = fixed["z_layers"]
        print(f"\n  [OK] Usando capas corregidas ({fixed['n_geo_conflicts']} cruces resueltos)")
    else:
        z_layers = compute_z_layers(g.edges)
        print(f"\n  [INFO] Usando capas básicas (ejecuta check_and_fix_collisions.py primero)")


    node_bars = {n['id']: [] for n in g.nodes}
    for i, e in enumerate(g.edges):
        node_bars[e['n1']].append(i)
        node_bars[e['n2']].append(i)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── BARRAS ───────────────────────────────────────────────────
    print(f"\n[BARRAS]  (Generando SOLO Barra 11 ESCALADA)")
    bar_names = []
    for i, e in enumerate(g.edges):
        if i != 11:
            continue
            
        L   = e['length'] * SCALE
        z   = z_layers[i] * LAYER_PITCH
        n1, n2 = e['n1'], e['n2']

        # Factor de escala para llegar a diámetro de hueco 6.0mm
        # Original HOLE_R es 2.4mm (diámetro 4.8mm)
        # Nuevo diámetro 6.0mm -> Nuevo HOLE_R = 3.0mm
        # Factor = 3.0 / 2.4 = 1.25
        FACTOR = 1.25

        L_scaled = L * FACTOR
        WIDTH_scaled = BAR_WIDTH * FACTOR
        THICK_scaled = BAR_THICK * FACTOR
        HOLE_R_scaled = HOLE_R * FACTOR
        
        print(f"\n  Dimensiones originales:")
        print(f"  - Longitud (L): {L:.2f} mm")
        print(f"  - Ancho exterior: {BAR_WIDTH:.2f} mm")
        print(f"  - Espesor Z: {BAR_THICK:.2f} mm")
        print(f"  - Diámetro de hueco: {HOLE_R * 2:.2f} mm")
        
        print(f"\n  Nuevas dimensiones (Escala {FACTOR}x):")
        print(f"  - Longitud (L): {L_scaled:.2f} mm")
        print(f"  - Ancho exterior: {WIDTH_scaled:.2f} mm")
        print(f"  - Espesor Z: {THICK_scaled:.2f} mm")
        print(f"  - Diámetro de hueco: {HOLE_R_scaled * 2:.2f} mm")

        bar = make_bar(L_scaled, WIDTH_scaled, THICK_scaled, HOLE_R_scaled)
        bar.apply_translation([0, 0, z])   # posicionar en su capa

        # Nombre actualizado
        fname = f"BARRA_{i:02d}__N{n1}-N{n2}__{L_scaled:.0f}mm__capa{z_layers[i]:+d}_ESCALADA_0.6cm.stl"
        bar.export(os.path.join(OUT_DIR, fname))
        bar_names.append(fname)
        print(f"  Exportado: {fname}")

    # ── PINES ────────────────────────────────────────────────────
    print(f"\n[PINES / ARTICULACIONES]  (Saltando pines)")
    pin_names = []
    for n in g.nodes:
        pass

    # ── Resumen ──────────────────────────────────────────────────
    all_stl = bar_names + pin_names
    total_kb = sum(
        os.path.getsize(os.path.join(OUT_DIR, f)) for f in all_stl
    ) / 1024

    print(f"\n{'='*60}")
    print(f"  ✓ {len(all_stl)} STLs en: {OUT_DIR}")
    print(f"  Barras: {len(bar_names)}   Pines: {len(pin_names)}")
    print(f"  Tamaño total: {total_kb:.0f} KB")
    print(f"\n  PARA IMPRIMIR:")
    print(f"  1. Importa los STL en tu slicer (PrusaSlicer / Cura).")
    print(f"  2. Barras: imprimir planas (Z=altura=BAR_THICK={BAR_THICK:.1f}mm).")
    print(f"     El hueco en cada extremo ya está incluido (HOLE_R={HOLE_R:.1f}mm).")
    print(f"  3. Pines: imprimir verticales (eje Z = eje del cilindro).")
    print(f"     PIN_R={PIN_R:.1f}mm — insertar con {HOLE_R-PIN_R:.1f}mm de holgura.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
