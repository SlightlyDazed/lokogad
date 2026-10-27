#!/usr/bin/env python3
"""
render_snapshots.py
Genera imágenes (renders 3D) del mecanismo de Theo Jansen a distintos
ángulos de la manivela utilizando la biblioteca trimesh y pyrender (offscreen).

Ejecutar: source ~/pytorch-rocm/bin/activate && python3 render_snapshots.py
"""
import os
import math
import numpy as np
import trimesh

# Habilitar render offscreen para pyrender (EGL)
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Importar constantes y funciones del script existente
import generate_jansen_urdf as jansen

OUTPUT_DIR = os.path.join(jansen.OUTPUT_DIR, "snapshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

colors = {
    "chassis": [140, 140, 150, 255], "crank_AC": [255, 70, 70, 255],
    "link_CD": [70, 200, 70, 255],   "link_BD": [70, 70, 220, 255],
    "link_BE": [220, 200, 50, 255],  "link_CE": [200, 70, 200, 255],
    "link_DF": [70, 200, 200, 255],  "link_BF": [140, 140, 210, 255],
    "link_FG": [220, 140, 50, 255],  "link_EG": [50, 140, 220, 255],
    "link_GH": [230, 50, 130, 255],  "link_EH": [50, 230, 130, 255],
}

def load_meshes():
    meshes = {}
    for name, _, _, _ in jansen.LINK_DEFS:
        p = os.path.join(jansen.STL_DIR, f"{name}.stl")
        if os.path.exists(p):
            m = trimesh.load(p)
            m.visual.face_colors = colors.get(name, [180, 180, 180, 255])
            m.apply_scale(0.001)
            meshes[name] = m
    return meshes

def bar_transform(p1, p2, name):
    sx, sy = p1[0]/1000, p1[1]/1000
    ex, ey = p2[0]/1000, p2[1]/1000
    angle = math.atan2(ey - sy, ex - sx)
    z = jansen.Z_LAYER.get(name, 0) * jansen.LAYER_PITCH / 1000
    T = np.eye(4)
    T[0, 0], T[0, 1] = math.cos(angle), -math.sin(angle)
    T[1, 0], T[1, 1] = math.sin(angle),  math.cos(angle)
    T[0, 3], T[1, 3], T[2, 3] = sx, sy, z
    return T

def pin_transform(pt, pname):
    bars = jansen.PIVOTS[pname]
    layers = [jansen.Z_LAYER[b] for b in bars if b in jansen.Z_LAYER]
    zmin = min(layers) * jansen.LAYER_PITCH / 1000 - jansen.BAR_THICK / 2000
    zmax = max(layers) * jansen.LAYER_PITCH / 1000 + jansen.BAR_THICK / 2000
    T = np.eye(4)
    T[0, 3] = pt[0] / 1000
    T[1, 3] = pt[1] / 1000
    T[2, 3] = (zmin + zmax) / 2
    return T

def capture_snapshots():
    print("[INFO] Cargando mallas...")
    meshes = load_meshes()
    
    # Suelo estático
    ground = trimesh.creation.box(extents=[0.2, 0.001, 0.06])
    ground.visual.face_colors = [40, 180, 40, 60]
    # Calculamos el nivel de suelo aproximado
    foot_pts = [jansen.solve(a).get('H') for a in range(360)]
    fy = [p[1]/1000 for p in foot_pts if p is not None]
    ground_y = min(fy) if fy else -0.092
    Tg = np.eye(4); Tg[1, 3] = ground_y
    ground.apply_transform(Tg)
    
    pin_radius = jansen.PIN_R * 0.001 * 1.2
    
    angles = [0, 90, 180, 270]
    for angle in angles:
        print(f"[INFO] Renderizando frame en ángulo: {angle}°...")
        scene = trimesh.Scene()
        scene.add_geometry(ground.copy(), node_name="ground", geom_name="ground")
        
        pts = jansen.solve(angle)
        if pts.get('H') is None:
            print(f"[WARN] No se pudo resolver la cinemática para {angle}° (NaN).")
            continue
            
        # Añadir barras transformadas
        for name, p1n, p2n, _ in jansen.LINK_DEFS:
            p1, p2 = pts.get(p1n), pts.get(p2n)
            if p1 and p2 and name in meshes:
                m = meshes[name].copy()
                m.apply_transform(bar_transform(p1, p2, name))
                scene.add_geometry(m, node_name=name, geom_name=name)
                
        # Añadir pines
        for pname in jansen.PIVOTS:
            pt = pts.get(pname)
            if pt:
                bars = jansen.PIVOTS[pname]
                layers = [jansen.Z_LAYER[b] for b in bars if b in jansen.Z_LAYER]
                if not layers:
                    continue
                zmin = min(layers) * jansen.LAYER_PITCH / 1000 - jansen.BAR_THICK / 2000
                zmax = max(layers) * jansen.LAYER_PITCH / 1000 + jansen.BAR_THICK / 2000
                height = zmax - zmin
                pin = trimesh.creation.cylinder(radius=pin_radius, height=height, sections=16)
                pin.visual.face_colors = [255, 220, 50, 230]
                pin.apply_transform(pin_transform(pt, pname))
                gname = f"pin_{pname}"
                scene.add_geometry(pin, node_name=gname, geom_name=gname)
        
        # Configurar la cámara para que mire al mecanismo
        # El mecanismo suele estar entre -0.1 y 0.1 en X y Y
        scene.set_camera(angles=(math.radians(70), math.radians(0), math.radians(30)), 
                         distance=0.4, 
                         center=[0.0, -0.05, 0.0])
        
        out_path = os.path.join(OUTPUT_DIR, f"jansen_urdf_angle_{angle:03d}.png")
        try:
            png_data = scene.save_image(resolution=(1280, 720), visible=True)
            with open(out_path, 'wb') as f:
                f.write(png_data)
            print(f"  [OK] Guardado: {out_path}")
        except Exception as e:
            print(f"  [ERROR] Al renderizar la imagen: {e}")
            print("  Asegúrate de que 'pyrender' esté instalado: pip install pyrender")

if __name__ == "__main__":
    capture_snapshots()
