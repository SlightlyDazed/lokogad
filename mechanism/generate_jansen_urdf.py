#!/usr/bin/env python3
"""
generate_jansen_urdf.py — Theo Jansen → STL + URDF + animación con pivotes visibles.

Cada barra se posiciona en una capa Z distinta. Cilindros de pivote
atraviesan las capas en cada punto (A-H) conectando visualmente las barras.

Ejecutar: source ~/pytorch-rocm/bin/activate && python3 generate_jansen_urdf.py
"""
import os, math, time
import numpy as np
import cadquery as cq

OUTPUT_DIR = os.path.expanduser("~/opensim_gait_data/theo_jansen")
STL_DIR = os.path.join(OUTPUT_DIR, "meshes")
os.makedirs(STL_DIR, exist_ok=True)

# === Mecanismo de Jansen ===
MN = [38, 7.8, 15, 50, 41.5, 39.3, 61.9, 55.8, 40.1, 39.4, 36.7, 65.7, 49]

LINK_DEFS = [
    ("chassis",  "A", "B", None),
    ("crank_AC", "A", "C", 2),
    ("link_CD",  "C", "D", 3),
    ("link_BD",  "B", "D", 4),
    ("link_BE",  "B", "E", 5),
    ("link_CE",  "C", "E", 6),
    ("link_DF",  "D", "F", 7),
    ("link_BF",  "B", "F", 8),
    ("link_FG",  "F", "G", 9),
    ("link_EG",  "E", "G", 10),
    ("link_GH",  "G", "H", 11),
    ("link_EH",  "E", "H", 12),
]

BAR_WIDTH = 5.0
BAR_THICK = 2.0
PIN_R = 1.5
LAYER_PITCH = 3.0  # mm entre capas

# Capas Z: barras se alternan a ambos lados del plano central (z=0)
# Garantiza que ningún par de barras que comparten pivote estén en la misma capa.
Z_LAYER = {
    "chassis":  0,
    "crank_AC": 1,
    "link_CD":  2,   "link_BD": -1,
    "link_CE": -2,   "link_BE":  3,
    "link_DF":  4,   "link_BF": -3,
    "link_FG":  5,   "link_EG": -4,
    "link_GH":  3,   "link_EH": -1,
}

# Qué barras conecta cada pivote (para generar pines)
PIVOTS = {
    "A": ["chassis", "crank_AC"],
    "B": ["chassis", "link_BD", "link_BE", "link_BF"],
    "C": ["crank_AC", "link_CD", "link_CE"],
    "D": ["link_CD", "link_BD", "link_DF"],
    "E": ["link_BE", "link_CE", "link_EG", "link_EH"],
    "F": ["link_DF", "link_BF", "link_FG"],
    "G": ["link_FG", "link_EG", "link_GH"],
    "H": ["link_GH", "link_EH"],
}


def linkage(p1, l1, p2, l2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    d = math.sqrt(dx**2 + dy**2)
    if d > l1+l2 or d < abs(l1-l2) or d == 0:
        return None
    a = (l1**2 - l2**2 + d**2) / (2*d)
    h = math.sqrt(max(0, l1**2 - a**2))
    mx, my = p1[0]+dx*a/d, p1[1]+dy*a/d
    rx, ry = -dy*h/d, dx*h/d
    r1, r2 = (mx+rx, my+ry), (mx-rx, my-ry)
    cross = (p2[0]-p1[0])*(r1[1]-p1[1]) - (p2[1]-p1[1])*(r1[0]-p1[0])
    return r1 if cross < 0 else r2


def solve(angle_deg):
    A, B = (0.0, 0.0), (-MN[0], -MN[1])
    rad = math.radians(angle_deg)
    C = (MN[2]*math.cos(rad), MN[2]*math.sin(rad))
    D = linkage(C, MN[3], B, MN[4])
    E = linkage(B, MN[5], C, MN[6])
    F = linkage(D, MN[7], B, MN[8]) if D else None
    G = linkage(F, MN[9], E, MN[10]) if (F and E) else None
    H = linkage(G, MN[11], E, MN[12]) if (G and E) else None
    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H}


# === STL generation ===
def gen_bar(name, length):
    hw, ht = BAR_WIDTH/2, BAR_THICK/2
    bar = (cq.Workplane("XY")
           .moveTo(0, -hw).lineTo(length, -hw)
           .threePointArc((length+hw, 0), (length, hw))
           .lineTo(0, hw)
           .threePointArc((-hw, 0), (0, -hw))
           .close().extrude(BAR_THICK).translate((0, 0, -ht)))
    bar = (bar.faces(">Z").workplane()
           .pushPoints([(0,0), (length,0)]).circle(PIN_R).cutThruAll())
    for ext in ['stl', 'step']:
        exp_type = cq.exporters.ExportTypes.STEP if ext == 'step' else cq.exporters.ExportTypes.STL
        cq.exporters.export(bar, os.path.join(STL_DIR, f"{name}.{ext}"),
                            exp_type if ext == 'step' else None)
    return length


def gen_all_stls():
    print("\n[INFO] Generando STLs...")
    lengths = {}
    for name, _, _, mi in LINK_DEFS:
        length = MN[mi] if mi is not None else math.sqrt(MN[0]**2 + MN[1]**2)
        lengths[name] = length
        stl_path = os.path.join(STL_DIR, f"{name}.stl")
        step_path = os.path.join(STL_DIR, f"{name}.step")
        hw, ht = BAR_WIDTH/2, BAR_THICK/2
        bar = (cq.Workplane("XY")
               .moveTo(0, -hw).lineTo(length, -hw)
               .threePointArc((length+hw, 0), (length, hw))
               .lineTo(0, hw)
               .threePointArc((-hw, 0), (0, -hw))
               .close().extrude(BAR_THICK).translate((0, 0, -ht)))
        bar = (bar.faces(">Z").workplane()
               .pushPoints([(0,0), (length,0)]).circle(PIN_R).cutThruAll())
        cq.exporters.export(bar, stl_path)
        cq.exporters.export(bar, step_path, cq.exporters.ExportTypes.STEP)
        print(f"  [OK] {name:.<20s} L={length:5.1f}mm")
    return lengths


# === Animación ===
def run_animation():
    import trimesh
    from trimesh.viewer.windowed import SceneViewer

    print("\n[INFO] Preparando animación con pivotes...")
    colors = {
        "chassis": [140,140,150,255], "crank_AC": [255,70,70,255],
        "link_CD": [70,200,70,255],   "link_BD": [70,70,220,255],
        "link_BE": [220,200,50,255],  "link_CE": [200,70,200,255],
        "link_DF": [70,200,200,255],  "link_BF": [140,140,210,255],
        "link_FG": [220,140,50,255],  "link_EG": [50,140,220,255],
        "link_GH": [230,50,130,255],  "link_EH": [50,230,130,255],
    }

    # Cargar meshes
    meshes = {}
    for name, _, _, _ in LINK_DEFS:
        p = os.path.join(STL_DIR, f"{name}.stl")
        if os.path.exists(p):
            m = trimesh.load(p)
            m.visual.face_colors = colors.get(name, [180,180,180,255])
            m.apply_scale(0.001)
            meshes[name] = m

    # Trayectoria del pie
    foot_pts = [solve(a).get('H') for a in range(360)]
    foot_pts = [p for p in foot_pts if p is not None]
    fx = [p[0]/1000 for p in foot_pts]
    fy = [p[1]/1000 for p in foot_pts]

    scene = trimesh.Scene()

    # Suelo
    ground_y = min(fy) if fy else -0.092
    ground = trimesh.creation.box(extents=[0.2, 0.001, 0.06])
    ground.visual.face_colors = [40, 180, 40, 60]
    Tg = np.eye(4); Tg[1, 3] = ground_y
    ground.apply_transform(Tg)
    scene.add_geometry(ground, node_name="ground", geom_name="ground")

    # Trayectoria del pie
    foot_z = Z_LAYER.get('link_GH', 0) * LAYER_PITCH / 1000
    pts3d = np.array([[x, y, foot_z] for x, y in zip(fx, fy)])
    if len(pts3d) > 2:
        scene.add_geometry(trimesh.load_path(pts3d),
                           node_name="foot_trace", geom_name="foot_trace")

    # Añadir barras
    for name in meshes:
        scene.add_geometry(meshes[name].copy(), node_name=name, geom_name=name)

    # Crear cilindros de pivote para cada punto (A-H)
    # Se crean como geometría dinámica que se reposiciona en cada frame
    pin_radius = PIN_R * 0.001 * 1.2  # ligeramente más grueso que el agujero
    for pname, bars in PIVOTS.items():
        layers = [Z_LAYER[b] for b in bars if b in Z_LAYER]
        if not layers:
            continue
        zmin = min(layers) * LAYER_PITCH / 1000 - BAR_THICK / 2000
        zmax = max(layers) * LAYER_PITCH / 1000 + BAR_THICK / 2000
        height = zmax - zmin
        pin = trimesh.creation.cylinder(radius=pin_radius, height=height, sections=16)
        pin.visual.face_colors = [255, 220, 50, 230]
        # Posición inicial (se actualiza en callback)
        Tp = np.eye(4)
        Tp[2, 3] = (zmin + zmax) / 2
        pin.apply_transform(Tp)
        gname = f"pin_{pname}"
        scene.add_geometry(pin, node_name=gname, geom_name=gname)

    _state = {'start': None, 'speed': 60.0}

    def bar_transform(p1, p2, name):
        sx, sy = p1[0]/1000, p1[1]/1000
        ex, ey = p2[0]/1000, p2[1]/1000
        angle = math.atan2(ey - sy, ex - sx)
        z = Z_LAYER.get(name, 0) * LAYER_PITCH / 1000
        T = np.eye(4)
        T[0,0], T[0,1] = math.cos(angle), -math.sin(angle)
        T[1,0], T[1,1] = math.sin(angle),  math.cos(angle)
        T[0,3], T[1,3], T[2,3] = sx, sy, z
        return T

    def pin_transform(pt, pname):
        """Posiciona un pin en el punto pivote."""
        bars = PIVOTS[pname]
        layers = [Z_LAYER[b] for b in bars if b in Z_LAYER]
        zmin = min(layers) * LAYER_PITCH / 1000 - BAR_THICK / 2000
        zmax = max(layers) * LAYER_PITCH / 1000 + BAR_THICK / 2000
        T = np.eye(4)
        T[0, 3] = pt[0] / 1000
        T[1, 3] = pt[1] / 1000
        T[2, 3] = (zmin + zmax) / 2
        return T

    def animate(scene):
        if _state['start'] is None:
            _state['start'] = time.time()
        t = time.time() - _state['start']
        angle = (t * _state['speed']) % 360
        pts = solve(angle)
        if pts.get('H') is None:
            return

        # Actualizar barras
        for name, p1n, p2n, _ in LINK_DEFS:
            p1, p2 = pts.get(p1n), pts.get(p2n)
            if p1 and p2:
                scene.graph.update(name, matrix=bar_transform(p1, p2, name))

        # Actualizar pines en los pivotes
        for pname in PIVOTS:
            pt = pts.get(pname)
            if pt:
                scene.graph.update(f"pin_{pname}", matrix=pin_transform(pt, pname))

    print(f"[OK] {len(meshes)} barras + {len(PIVOTS)} pivotes")
    print("[INFO] Abriendo animación... (cierra la ventana para terminar)")
    print("       Arrastra=rotar, scroll=zoom, shift+drag=pan")

    SceneViewer(scene, callback=animate, callback_period=1/30,
                resolution=(1280, 720),
                window_title="Theo Jansen — Mecanismo con pivotes")


# === URDF ===
def gen_urdf(lengths):
    path = os.path.join(OUTPUT_DIR, "theo_jansen.urdf")
    colors = {
        "chassis": "0.5 0.5 0.5", "crank_AC": "1 0.3 0.3",
        "link_CD": "0.3 0.8 0.3", "link_BD": "0.3 0.3 0.8",
        "link_BE": "0.8 0.8 0.3", "link_CE": "0.8 0.3 0.8",
        "link_DF": "0.3 0.8 0.8", "link_BF": "0.5 0.5 0.8",
        "link_FG": "0.8 0.5 0.3", "link_EG": "0.3 0.5 0.8",
        "link_GH": "0.9 0.2 0.5", "link_EH": "0.2 0.9 0.5",
    }
    xml = '<?xml version="1.0"?>\n<robot name="theo_jansen_leg">\n'
    for name, _, _, _ in LINK_DEFS:
        L = lengths[name]
        z = Z_LAYER.get(name, 0) * LAYER_PITCH / 1000
        c = colors.get(name, "0.5 0.5 0.5")
        xml += f'''
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="meshes/{name}.stl" scale="0.001 0.001 0.001"/></geometry>
      <material name="m_{name}"><color rgba="{c} 1.0"/></material>
    </visual>
    <inertial>
      <mass value="0.005"/>
      <origin xyz="{L/2000:.4f} 0 0"/>
      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>
    </inertial>
  </link>\n'''

    # Tree joints with Z-layer offsets
    dz = lambda n: Z_LAYER.get(n, 0) * LAYER_PITCH / 1000
    def jxml(jn, par, child, xyz):
        xml = f'  <joint name="{jn}" type="revolute">\n'
        xml += f'    <parent link="{par}"/><child link="{child}"/>\n'
        xml += f'    <origin xyz="{xyz}" rpy="0 0 0"/>\n'
        xml += f'    <axis xyz="0 0 1"/>\n'
        xml += f'    <limit lower="-6.28" upper="6.28" effort="1" velocity="6.28"/>\n'
        xml += f'  </joint>\n'
        return xml

    bx, by = -MN[0]/1000, -MN[1]/1000
    cl = lengths['crank_AC']/1000

    xml += jxml("j_crank",  "chassis",  "crank_AC", f"0 0 {dz('crank_AC')-dz('chassis'):.4f}")
    xml += jxml("j_C_CD",   "crank_AC", "link_CD",  f"{cl:.4f} 0 {dz('link_CD')-dz('crank_AC'):.4f}")
    xml += jxml("j_B_BD",   "chassis",  "link_BD",  f"{bx:.4f} {by:.4f} {dz('link_BD')-dz('chassis'):.4f}")
    xml += jxml("j_C_CE",   "crank_AC", "link_CE",  f"{cl:.4f} 0 {dz('link_CE')-dz('crank_AC'):.4f}")
    xml += jxml("j_B_BE",   "chassis",  "link_BE",  f"{bx:.4f} {by:.4f} {dz('link_BE')-dz('chassis'):.4f}")
    xml += jxml("j_D_DF",   "link_CD",  "link_DF",  f"{lengths['link_CD']/1000:.4f} 0 {dz('link_DF')-dz('link_CD'):.4f}")
    xml += jxml("j_B_BF",   "chassis",  "link_BF",  f"{bx:.4f} {by:.4f} {dz('link_BF')-dz('chassis'):.4f}")
    xml += jxml("j_F_FG",   "link_DF",  "link_FG",  f"{lengths['link_DF']/1000:.4f} 0 {dz('link_FG')-dz('link_DF'):.4f}")
    xml += jxml("j_E_EG",   "link_BE",  "link_EG",  f"{lengths['link_BE']/1000:.4f} 0 {dz('link_EG')-dz('link_BE'):.4f}")
    xml += jxml("j_G_GH",   "link_FG",  "link_GH",  f"{lengths['link_FG']/1000:.4f} 0 {dz('link_GH')-dz('link_FG'):.4f}")
    xml += jxml("j_E_EH",   "link_CE",  "link_EH",  f"{lengths['link_CE']/1000:.4f} 0 {dz('link_EH')-dz('link_CE'):.4f}")

    xml += '\n</robot>\n'
    with open(path, 'w') as f:
        f.write(xml)
    print(f"\n[OK] URDF: {path}")


# === Main ===
def main():
    print("=" * 60)
    print("  Theo Jansen → STL + URDF + Animación con pivotes")
    print("=" * 60)
    lengths = gen_all_stls()
    gen_urdf(lengths)
    print(f"\n{'='*60}")
    for root, _, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            if f.endswith(('.stl','.step','.urdf')):
                fp = os.path.join(root, f)
                print(f"  {os.path.relpath(fp, OUTPUT_DIR):<35s} {os.path.getsize(fp):>8,d} B")
    print(f"{'='*60}")
    run_animation()

if __name__ == "__main__":
    main()
