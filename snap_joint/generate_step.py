#!/usr/bin/env python3
"""
generate_step.py — Genera archivos STEP del snap pin joint usando CadQuery.
Geometría paramétrica real (no malla triangulada) — ideal para Fusion 360.
Ejecutar: source ~/pytorch-rocm/bin/activate && python3 generate_step.py
"""
import os
import cadquery as cq

OUTPUT_DIR = os.path.expanduser("~/opensim_gait_data/gad_joints")

# === Dimensiones (mm) — idénticas al STL ===
LINK_LEN = 60.0
LINK_W = 12.0
LINK_H = 4.0
PIN_D = 7.0        # diámetro shaft
PIN_H = 5.0        # altura total del pin
RIDGE_D = 8.6      # diámetro reborde
RIDGE_H = 1.0      # altura reborde
HOLE_D = 7.3       # diámetro agujero
CAV_D = 8.7        # diámetro cavidad
CAV_H = 1.5        # altura cavidad
PIN_X = 54.0       # posición X del pin
PIN_Y = 0.0        # posición Y del pin (centro del eslabón)

# =========================================================================
#  Link A — eslabón con pin cilíndrico + reborde snap
# =========================================================================
def build_link_a():
    # Cuerpo rectangular
    body = (
        cq.Workplane("XY")
        .box(LINK_LEN, LINK_W, LINK_H, centered=False)
    )

    # Pin shaft: cilindro sobre el cuerpo
    shaft_h = PIN_H - RIDGE_H  # 4mm de shaft puro
    pin = (
        cq.Workplane("XY")
        .transformed(offset=(PIN_X, LINK_W / 2, LINK_H))
        .circle(PIN_D / 2)
        .extrude(shaft_h)
    )

    # Reborde snap: cilindro más ancho en la punta
    ridge = (
        cq.Workplane("XY")
        .transformed(offset=(PIN_X, LINK_W / 2, LINK_H + shaft_h))
        .circle(RIDGE_D / 2)
        .extrude(RIDGE_H)
    )

    result = body.union(pin).union(ridge)
    return result


# =========================================================================
#  Link B — eslabón con agujero cilíndrico + cavidad
# =========================================================================
def build_link_b():
    # Cuerpo rectangular
    body = (
        cq.Workplane("XY")
        .box(LINK_LEN, LINK_W, LINK_H, centered=False)
    )

    # Agujero pasante (diámetro 7.3mm) — parte inferior
    hole_lower_h = LINK_H - CAV_H  # 2.5mm
    hole_lower = (
        cq.Workplane("XY")
        .transformed(offset=(PIN_X, LINK_W / 2, 0))
        .circle(HOLE_D / 2)
        .extrude(hole_lower_h)
    )

    # Cavidad superior (diámetro 8.7mm) — para recibir reborde
    cavity = (
        cq.Workplane("XY")
        .transformed(offset=(PIN_X, LINK_W / 2, hole_lower_h))
        .circle(CAV_D / 2)
        .extrude(CAV_H)
    )

    result = body.cut(hole_lower).cut(cavity)
    return result


# =========================================================================
#  Main
# =========================================================================
def main():
    print("=" * 60)
    print("  GAD — Generando archivos STEP (CadQuery)")
    print("=" * 60)

    # Link A
    print("[INFO] Construyendo Link A (pin + reborde)...")
    link_a = build_link_a()
    path_a = os.path.join(OUTPUT_DIR, "link_A.step")
    cq.exporters.export(link_a, path_a, cq.exporters.ExportTypes.STEP)
    print(f"[OK] {path_a} ({os.path.getsize(path_a):,d} bytes)")

    # Link B
    print("[INFO] Construyendo Link B (agujero + cavidad)...")
    link_b = build_link_b()
    path_b = os.path.join(OUTPUT_DIR, "link_B.step")
    cq.exporters.export(link_b, path_b, cq.exporters.ExportTypes.STEP)
    print(f"[OK] {path_b} ({os.path.getsize(path_b):,d} bytes)")

    # Ensamblaje completo (ambos links en un solo STEP)
    print("[INFO] Construyendo ensamblaje...")
    assembly = cq.Assembly()
    assembly.add(link_a, name="link_A", color=cq.Color(0.9, 0.9, 0.95, 1.0))
    # Link B trasladado en Z para simular ensamblaje
    assembly.add(link_b, name="link_B",
                 loc=cq.Location((0, 0, LINK_H)),
                 color=cq.Color(0.2, 0.4, 0.85, 1.0))
    path_asm = os.path.join(OUTPUT_DIR, "snap_joint_assembly.step")
    assembly.save(path_asm)
    print(f"[OK] {path_asm} ({os.path.getsize(path_asm):,d} bytes)")

    print(f"\n{'='*60}")
    print("  Archivos STEP generados:")
    print(f"{'='*60}")
    for f in ["link_A.step", "link_B.step", "snap_joint_assembly.step"]:
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(fp):
            print(f"  {f:<35s} {os.path.getsize(fp):>8,d} bytes")

    print(f"\nImportar en Fusion 360: File → Open → seleccionar .step")


if __name__ == "__main__":
    main()
