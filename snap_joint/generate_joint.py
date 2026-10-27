#!/usr/bin/env python3
"""
generate_joint.py — Genera STL + URDF + GAD node para snap pin joint.
Geometría: dos eslabones 60×12×4mm conectados por un pin a presión.
"""
import math
import os
import json

OUTPUT_DIR = os.path.expanduser("~/opensim_gait_data/gad_joints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N = 32  # segmentos para cilindros

# === Dimensiones (mm) ===
LINK_LEN, LINK_W, LINK_H = 60.0, 12.0, 4.0
PIN_R = 3.5       # radio shaft (diám 7mm)
PIN_H = 5.0       # altura total del pin
RIDGE_R = 4.3     # radio reborde (diám 8.6mm)
RIDGE_H = 1.0     # altura reborde
HOLE_R = 3.65     # radio agujero (diám 7.3mm)
CAV_R = 4.35      # radio cavidad (diám 8.7mm)
CAV_H = 1.5       # altura cavidad
PIN_CX, PIN_CY = 54.0, 0.0  # posición del pin en el eslabón

# =========================================================================
#  Helpers STL
# =========================================================================
def norm(v):
    l = math.sqrt(sum(c*c for c in v))
    return tuple(c/l for c in v) if l > 1e-12 else (0,0,1)

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def face_normal(v1, v2, v3):
    e1 = tuple(b-a for a,b in zip(v1, v2))
    e2 = tuple(b-a for a,b in zip(v1, v3))
    return norm(cross(e1, e2))

def tri_stl(v1, v2, v3):
    n = face_normal(v1, v2, v3)
    def vf(v): return f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}"
    return (f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n"
            f"    outer loop\n{vf(v1)}\n{vf(v2)}\n{vf(v3)}\n"
            f"    endloop\n  endfacet")

def write_stl(path, name, tris):
    with open(path, 'w') as f:
        f.write(f"solid {name}\n")
        for t in tris:
            f.write(tri_stl(*t) + "\n")
        f.write(f"endsolid {name}\n")
    print(f"[OK] {path} — {len(tris)} triángulos")

# =========================================================================
#  Geometría primitiva
# =========================================================================
def circle_pts(cx, cy, r, z):
    return [(cx + r*math.cos(2*math.pi*i/N),
             cy + r*math.sin(2*math.pi*i/N), z) for i in range(N)]

def box_tris(x0, y0, z0, x1, y1, z1):
    """Caja cerrada sin agujeros."""
    c = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
         (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
    faces = [
        (0,2,1),(0,3,2),  # bottom -Z
        (4,5,6),(4,6,7),  # top +Z
        (0,1,5),(0,5,4),  # front -Y
        (2,3,7),(2,7,6),  # back +Y
        (0,4,7),(0,7,3),  # left -X
        (1,2,6),(1,6,5),  # right +X
    ]
    return [(c[a],c[b],c[d]) for a,b,d in faces]

def cyl_wall(cx, cy, z0, z1, r, inward=False):
    """Pared cilíndrica."""
    b = circle_pts(cx, cy, r, z0)
    t = circle_pts(cx, cy, r, z1)
    tris = []
    for i in range(N):
        j = (i+1) % N
        if not inward:
            tris += [(b[i],t[i],t[j]), (b[i],t[j],b[j])]
        else:
            tris += [(b[i],t[j],t[i]), (b[i],b[j],t[j])]
    return tris

def disc_cap(cx, cy, r, z, up=True):
    """Tapa circular."""
    center = (cx, cy, z)
    pts = circle_pts(cx, cy, r, z)
    tris = []
    for i in range(N):
        j = (i+1) % N
        if up:
            tris.append((center, pts[i], pts[j]))
        else:
            tris.append((center, pts[j], pts[i]))
    return tris

def annulus(cx, cy, ri, ro, z, up=True):
    """Cara anular entre dos radios."""
    inner = circle_pts(cx, cy, ri, z)
    outer = circle_pts(cx, cy, ro, z)
    tris = []
    for i in range(N):
        j = (i+1) % N
        if up:
            tris += [(inner[i],outer[i],outer[j]), (inner[i],outer[j],inner[j])]
        else:
            tris += [(inner[i],outer[j],outer[i]), (inner[i],inner[j],outer[j])]
    return tris

def rect_with_hole(x0, y0, x1, y1, cx, cy, r, z, up=True):
    """Cara rectangular con agujero circular — triangulación por rayos."""
    tris = []
    # Puntos del agujero
    hole = circle_pts(cx, cy, r, z)
    # Para cada punto del agujero, proyectar al borde del rectángulo
    boundary = []
    corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    # Añadir esquinas con su ángulo
    corner_angles = []
    for (px,py) in corners:
        a = math.atan2(py - cy, px - cx)
        corner_angles.append(a)
        boundary.append((a, px, py, True))
    # Añadir proyecciones de puntos del circulo
    for i in range(N):
        a = 2*math.pi*i/N
        # Normalizar a [-pi, pi]
        if a > math.pi: a -= 2*math.pi
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        if abs(dx) > 1e-10:
            for xv in [x0, x1]:
                t = (xv - cx) / dx
                if t > 1e-6:
                    yv = cy + t*dy
                    if y0-0.01 <= yv <= y1+0.01:
                        ts.append(t)
        if abs(dy) > 1e-10:
            for yv in [y0, y1]:
                t = (yv - cy) / dy
                if t > 1e-6:
                    xv = cx + t*dx
                    if x0-0.01 <= xv <= x1+0.01:
                        ts.append(t)
        if ts:
            t = min(ts)
            boundary.append((a, cx+t*dx, cy+t*dy, False))

    boundary.sort(key=lambda x: x[0])
    nb = len(boundary)

    for i in range(nb):
        j = (i+1) % nb
        ai, aj = boundary[i][0], boundary[j][0]
        bi = (boundary[i][1], boundary[i][2], z)
        bj = (boundary[j][1], boundary[j][2], z)
        ci_ = (cx + r*math.cos(ai), cy + r*math.sin(ai), z)
        cj_ = (cx + r*math.cos(aj), cy + r*math.sin(aj), z)
        if up:
            tris += [(ci_, bi, bj), (ci_, bj, cj_)]
        else:
            tris += [(ci_, bj, bi), (ci_, cj_, bj)]
    return tris

# =========================================================================
#  Link A — eslabón con pin + reborde
# =========================================================================
def generate_link_a():
    tris = []
    x0, y0, z0 = 0, -LINK_W/2, 0
    x1, y1, z1 = LINK_LEN, LINK_W/2, LINK_H
    cx, cy = PIN_CX, PIN_CY
    shaft_top = z1 + PIN_H - RIDGE_H  # z=8
    ridge_top = z1 + PIN_H            # z=9

    # Caja: fondo (solid), techo (con agujero), 4 paredes
    # Fondo z=0
    tris += [(( x0,y0,z0),(x1,y0,z0),(x1,y1,z0)),
             ((x0,y0,z0),(x1,y1,z0),(x0,y1,z0))]
    # Techo z=4 con agujero para pin
    tris += rect_with_hole(x0, y0, x1, y1, cx, cy, PIN_R, z1, up=True)
    # 4 paredes laterales
    for (a,b,c,d) in [
        ((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)),  # frente -Y
        ((x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)),  # derecho +X
        ((x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1)),  # atrás +Y
        ((x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1)),  # izq -X
    ]:
        tris += [(a,b,c), (a,c,d)]

    # Pin shaft: pared cilíndrica r=3.5, z=4 a z=8
    tris += cyl_wall(cx, cy, z1, shaft_top, PIN_R)
    # Cara anular z=8: r=3.5 → r=4.3 (base del reborde)
    tris += annulus(cx, cy, PIN_R, RIDGE_R, shaft_top, up=True)
    # Reborde: pared cilíndrica r=4.3, z=8 a z=9
    tris += cyl_wall(cx, cy, shaft_top, ridge_top, RIDGE_R)
    # Tapa superior del reborde z=9
    tris += disc_cap(cx, cy, RIDGE_R, ridge_top, up=True)

    return tris

# =========================================================================
#  Link B — eslabón con agujero + cavidad
# =========================================================================
def generate_link_b():
    tris = []
    x0, y0, z0 = 0, -LINK_W/2, 0
    x1, y1, z1 = LINK_LEN, LINK_W/2, LINK_H
    cx, cy = PIN_CX, PIN_CY
    cav_z = z1 - CAV_H  # z=2.5 — inicio de la cavidad

    # Fondo z=0 con agujero r=3.65
    tris += rect_with_hole(x0, y0, x1, y1, cx, cy, HOLE_R, z0, up=False)
    # Techo z=4 con agujero r=4.35 (cavidad)
    tris += rect_with_hole(x0, y0, x1, y1, cx, cy, CAV_R, z1, up=True)
    # 4 paredes laterales (igual que link_a)
    for (a,b,c,d) in [
        ((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)),
        ((x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)),
        ((x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1)),
        ((x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1)),
    ]:
        tris += [(a,b,c), (a,c,d)]

    # Agujero inferior: pared cilíndrica r=3.65, z=0 a z=2.5 (inward)
    tris += cyl_wall(cx, cy, z0, cav_z, HOLE_R, inward=True)
    # Cara anular z=2.5: r=3.65 → r=4.35 (transición a cavidad)
    tris += annulus(cx, cy, HOLE_R, CAV_R, cav_z, up=True)
    # Cavidad superior: pared cilíndrica r=4.35, z=2.5 a z=4 (inward)
    tris += cyl_wall(cx, cy, cav_z, z1, CAV_R, inward=True)

    return tris

# =========================================================================
#  URDF
# =========================================================================
def generate_urdf():
    # Convertir mm a metros para URDF
    jx = PIN_CX / 1000.0  # 0.054
    jz = LINK_H / 1000.0 / 2.0  # centro del link en z

    urdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<robot name="snap_pin_joint" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- ============ Materials ============ -->
  <material name="pla_white">
    <color rgba="0.95 0.95 0.95 1.0"/>
  </material>
  <material name="pla_blue">
    <color rgba="0.2 0.4 0.8 1.0"/>
  </material>

  <!-- ============ Link A (con pin) ============ -->
  <link name="link_A">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="link_A.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="pla_white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="link_A.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <origin xyz="{LINK_LEN/2/1000:.4f} 0 {LINK_H/2/1000:.4f}" rpy="0 0 0"/>
      <inertia ixx="6.13e-07" ixy="0" ixz="0"
               iyy="3.01e-06" iyz="0" izz="3.49e-06"/>
    </inertial>
  </link>

  <!-- ============ Link B (con agujero) ============ -->
  <link name="link_B">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="link_B.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="pla_blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="link_B.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <origin xyz="{LINK_LEN/2/1000:.4f} 0 {LINK_H/2/1000:.4f}" rpy="0 0 0"/>
      <inertia ixx="6.13e-07" ixy="0" ixz="0"
               iyy="3.01e-06" iyz="0" izz="3.49e-06"/>
    </inertial>
  </link>

  <!-- ============ Joint revolute ============ -->
  <joint name="snap_pin" type="revolute">
    <parent link="link_A"/>
    <child link="link_B"/>
    <origin xyz="{jx:.4f} 0 {LINK_H/1000:.4f}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="{-math.pi/2:.6f}" upper="{math.pi/2:.6f}"
           effort="1.0" velocity="3.14"/>
    <dynamics damping="0.001" friction="0.005"/>
  </joint>

</robot>
"""
    path = os.path.join(OUTPUT_DIR, "snap_joint.urdf")
    with open(path, 'w') as f:
        f.write(urdf)
    print(f"[OK] {path}")

# =========================================================================
#  GAD Node
# =========================================================================
def generate_gad_node():
    node = {
        "type": "snap_pin",
        "dof": 1,
        "axis": [0, 0, 1],
        "limits_deg": [-90, 90],
        "limits_rad": [round(-math.pi/2, 6), round(math.pi/2, 6)],
        "pin_diameter_mm": PIN_R * 2,
        "pin_height_mm": PIN_H,
        "ridge_diameter_mm": RIDGE_R * 2,
        "ridge_height_mm": RIDGE_H,
        "hole_diameter_mm": HOLE_R * 2,
        "cavity_diameter_mm": CAV_R * 2,
        "cavity_height_mm": CAV_H,
        "clearance_radial_mm": HOLE_R - PIN_R,
        "interference_ridge_mm": RIDGE_R - CAV_R,
        "link_dimensions_mm": [LINK_LEN, LINK_W, LINK_H],
        "material": "PLA",
        "mass_kg": 0.01,
        "joint_position_mm": [PIN_CX, PIN_CY, LINK_H],
        "stl_files": ["link_A.stl", "link_B.stl"],
        "urdf_file": "snap_joint.urdf",
        "printable": True,
        "assembly_method": "press_fit_snap",
        "min_wall_thickness_mm": LINK_LEN - PIN_CX - RIDGE_R,
    }

    py_content = f'''#!/usr/bin/env python3
"""GAD graph node definition for snap pin joint."""

import math

joint_node = {json.dumps(node, indent=4)}

if __name__ == "__main__":
    print("=== GAD Snap Pin Joint Node ===")
    for k, v in joint_node.items():
        print(f"  {{k:.<30s}} {{v}}")
'''
    path = os.path.join(OUTPUT_DIR, "gad_node.py")
    with open(path, 'w') as f:
        f.write(py_content)
    print(f"[OK] {path}")
    return node

# =========================================================================
#  Main
# =========================================================================
def main():
    print("=" * 60)
    print("  GAD — Generando Snap Pin Joint")
    print("=" * 60)

    # STL Link A
    tris_a = generate_link_a()
    write_stl(os.path.join(OUTPUT_DIR, "link_A.stl"), "link_A", tris_a)

    # STL Link B
    tris_b = generate_link_b()
    write_stl(os.path.join(OUTPUT_DIR, "link_B.stl"), "link_B", tris_b)

    # URDF
    generate_urdf()

    # GAD Node
    node = generate_gad_node()

    print(f"\n{'='*60}")
    print("  Archivos generados en:", OUTPUT_DIR)
    print(f"{'='*60}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            print(f"  {f:<25s} {sz:>8,d} bytes")

    print(f"\nGAD Node:")
    for k, v in node.items():
        print(f"  {k:.<30s} {v}")

if __name__ == "__main__":
    main()
