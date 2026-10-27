#!/usr/bin/env python3
"""
=========================================================
  BRAZO DE MOTOR — Pieza de extensión con dos hubs de 3mm
=========================================================
  Geometría (medidas borde-a-borde):
    - Extremo A (x=0): Hueco Ø 4.8mm + boquilla 15mm sobresaliente
    - 65mm borde-a-borde: desde borde der. hueco A → borde der. Hub2
    - Hub 1 (interior): Ø hueco 3mm, a 15mm borde-a-borde del Hub2
    - Hub 2 (final):    Ø hueco 3mm, en el extremo de los 65mm

  Parámetros (consistentes con barras redimensionadas 1.25x):
    BAR_WIDTH  = 7.5mm
    BAR_THICK  = 3.75mm
    R_A        = 2.4mm  → Ø 4.8mm (hueco estándar)
    R_hub      = 1.5mm  → Ø 3mm   (hubs de motor)
    BOSS_H     = 15.0mm → boquilla sobre extremo A
=========================================================
"""

import os
import numpy as np
import trimesh
from shapely.geometry import Point, box as shbox
import shapely.affinity

# ─── Parámetros ──────────────────────────────────────────
BAR_WIDTH = 7.5      # mm
BAR_THICK = 3.75     # mm
R_A       = 2.4      # mm  — radio hueco extremo A (Ø 4.8mm)
R_hub     = 1.5      # mm  — radio huecos hubs (Ø 3mm)
SPAN      = 65.0     # mm  — borde-a-borde: hueco A → Hub2
BOSS_H    = 15.0     # mm  — altura de la boquilla en extremo A
SECTIONS  = 64

# Radio exterior de la boquilla con pared DOBLE
# Pared original = hw - R_A = 3.75 - 2.4 = 1.35 mm
# Pared doble    = 2 × 1.35 = 2.70 mm  → boss_hw = R_A + 2*(hw - R_A)
# (hw se calcula abajo; boss_hw se define después de hw)

hw       = BAR_WIDTH / 2.0           # 3.75mm — radio exterior barra
boss_hw  = R_A + 2.0 * (hw - R_A)   # 5.10mm — radio exterior boquilla (pared x2)

# Centros X (origen = centro del hueco A)
xA    = 0.0
xHub2 = R_A + SPAN - R_hub          # 2.4 + 65 - 1.5 = 65.9 mm
xHub1 = xHub2 - (15.0 + 2 * R_hub) # 65.9 - 18 = 47.9 mm  (15mm borde-a-borde)

TOTAL = R_A + SPAN  # 67.4 mm — longitud total borde-a-borde

# ─── Directorio de salida ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repositorio
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "stl_for_print", "barras_redim_1.25x")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Perfil 2D del cuerpo principal ──────────────────────
outer_rect = shbox(xA, -hw, xHub2, hw)
cap_a      = Point(xA,    0).buffer(hw, resolution=SECTIONS)
cap_h2     = Point(xHub2, 0).buffer(hw, resolution=SECTIONS)
solid      = outer_rect.union(cap_a).union(cap_h2)

# Huecos
hole_a    = Point(xA,    0).buffer(R_A,   resolution=SECTIONS)
hole_hub1 = Point(xHub1, 0).buffer(R_hub, resolution=SECTIONS)
hole_hub2 = Point(xHub2, 0).buffer(R_hub, resolution=SECTIONS)

profile = solid.difference(hole_a).difference(hole_hub1).difference(hole_hub2)

# ─── Extrusión 3D del cuerpo ─────────────────────────────
mesh_body = trimesh.creation.extrude_polygon(profile, BAR_THICK)
mesh_body.apply_translation([0.0, 0.0, -BAR_THICK / 2.0])

# ─── Boquilla (boss) en extremo A — pared DOBLE, travesando barra entera ──
# La boquilla ATRAVIESA el grosor completo de la barra (máximo contacto lateral)
# → ya no es solo una cara plana, sino la pared a lo largo de BAR_THICK+BOSS_H
# Perfil: círculo exterior radio=boss_hw (pared x2) con hueco Ø 4.8mm
boss_outer   = Point(xA, 0).buffer(boss_hw, resolution=SECTIONS)
boss_profile = boss_outer.difference(hole_a)

boss_total_h = BAR_THICK + BOSS_H        # atraviesa barra + sobresale BOSS_H
mesh_boss = trimesh.creation.extrude_polygon(boss_profile, boss_total_h)
# Arranca desde la cara INFERIOR de la barra (z = -BAR_THICK/2)
# y termina en z = -BAR_THICK/2 + BAR_THICK + BOSS_H = BAR_THICK/2 + BOSS_H
mesh_boss.apply_translation([0.0, 0.0, -BAR_THICK / 2.0])

# ─── Unir todo ───────────────────────────────────────────
mesh_final = trimesh.util.concatenate([mesh_body, mesh_boss])

# ─── Exportar ────────────────────────────────────────────
fname = f"BRAZO_MOTOR__L{TOTAL:.0f}mm__hubs_d3mm__boquilla15mm.stl"
fpath = os.path.join(OUT_DIR, fname)
mesh_final.export(fpath)

print()
print("=" * 62)
print("  BRAZO DE MOTOR — STL Generado  [Boquilla pared x2]")
print("=" * 62)
print(f"  Longitud total (borde-a-borde) : {TOTAL:.1f} mm")
print(f"  Ancho barra                    : {BAR_WIDTH} mm")
print(f"  Espesor Z cuerpo               : {BAR_THICK} mm")
print(f"  Boquilla en extremo A          : {BOSS_H} mm de altura")
print(f"  Boquilla (oval)  semi_x        : {boss_hw:.2f} mm  (pared {boss_hw-R_A:.2f} mm = x2 original)")
print(f"  Boquilla (oval)  semi_y        : {hw:.2f} mm  (ancho barra, máximo contacto)")
print()
print(f"  Hueco extremo A (x={xA})       : Ø {R_A*2:.1f} mm (estándar pin)")
print(f"  Hub 1  (x={xHub1:.1f}mm)       : Ø {R_hub*2:.0f} mm  (interior)")
print(f"  Hub 2  (x={xHub2:.1f}mm)       : Ø {R_hub*2:.0f} mm  (final, borde=65mm)")
print(f"  Distancia borde-a-borde hubs   : {(xHub2-R_hub)-(xHub1+R_hub):.0f} mm")
print()
print(f"  Archivo: {fname}")
print(f"  Carpeta: {OUT_DIR}")
print("=" * 62)
