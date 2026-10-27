#!/usr/bin/env python3
"""
view_urdf.py — Visualiza el URDF del snap pin joint
Usa yourdfpy o trimesh para renderizar en ventana 3D.
Ejecutar con: source ~/pytorch-rocm/bin/activate && python3 view_urdf.py
"""
import os
import sys
import math

JOINT_DIR = os.path.expanduser("~/opensim_gait_data/gad_joints")
URDF_PATH = os.path.join(JOINT_DIR, "snap_joint.urdf")


def view_with_yourdfpy():
    """Visualizar URDF completo con yourdfpy."""
    import yourdfpy
    print("[INFO] Cargando URDF con yourdfpy...")
    urdf = yourdfpy.URDF.load(
        URDF_PATH,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )

    # Información del URDF
    links = list(urdf.link_map.keys()) if hasattr(urdf, 'link_map') else []
    joints = list(urdf.joint_map.keys()) if hasattr(urdf, 'joint_map') else []
    print(f"[OK] URDF cargado")
    print(f"     Links: {links}")
    print(f"     Joints: {joints}")

    # Configurar posición del joint (30°) para demostrar rotación
    if hasattr(urdf, 'cfg') and urdf.cfg is not None:
        import numpy as np
        urdf.cfg = np.array([math.radians(30)])
        urdf.update_cfg(urdf.cfg)

    print("[INFO] Abriendo visor 3D (cierra la ventana para terminar)...")
    urdf.show()


def view_with_trimesh():
    """Fallback: visualizar STLs con trimesh."""
    import trimesh
    import numpy as np

    print("[INFO] Cargando STLs con trimesh...")
    scene = trimesh.Scene()

    # Link A — color gris claro
    mesh_a = trimesh.load(os.path.join(JOINT_DIR, "link_A.stl"))
    mesh_a.visual.face_colors = [220, 220, 220, 255]
    mesh_a.apply_scale(0.001)  # mm → m
    scene.add_geometry(mesh_a, node_name="link_A")

    # Link B — color azul, en posición ensamblada con rotación 30°
    mesh_b = trimesh.load(os.path.join(JOINT_DIR, "link_B.stl"))
    mesh_b.visual.face_colors = [50, 100, 200, 255]
    mesh_b.apply_scale(0.001)

    # Posicionar: trasladar z +4mm (encima de link_A) y rotar 30° en Z
    jx, jz = 0.054, 0.004
    angle = math.radians(30)
    T = np.eye(4)
    T[2, 3] = jz  # desplazar en Z
    R = trimesh.transformations.rotation_matrix(angle, [0, 0, 1], [jx, 0, jz])
    mesh_b.apply_transform(R @ T)

    scene.add_geometry(mesh_b, node_name="link_B")

    print(f"[OK] Escena: {len(scene.geometry)} meshes")
    for name, geom in scene.geometry.items():
        print(f"     {name}: {len(geom.faces)} caras, bounds: {geom.bounds}")

    print("[INFO] Abriendo visor 3D (cierra la ventana para terminar)...")
    scene.show()


def export_screenshot():
    """Exportar render como imagen PNG (sin ventana)."""
    import trimesh
    import numpy as np

    print("[INFO] Generando render estático...")
    scene = trimesh.Scene()

    mesh_a = trimesh.load(os.path.join(JOINT_DIR, "link_A.stl"))
    mesh_a.visual.face_colors = [220, 220, 220, 255]
    mesh_a.apply_scale(0.001)
    scene.add_geometry(mesh_a, node_name="link_A")

    mesh_b = trimesh.load(os.path.join(JOINT_DIR, "link_B.stl"))
    mesh_b.visual.face_colors = [50, 100, 200, 255]
    mesh_b.apply_scale(0.001)
    jx, jz = 0.054, 0.004
    T = np.eye(4)
    T[2, 3] = jz
    angle = math.radians(30)
    R = trimesh.transformations.rotation_matrix(angle, [0, 0, 1], [jx, 0, jz])
    mesh_b.apply_transform(R @ T)
    scene.add_geometry(mesh_b, node_name="link_B")

    out = os.path.join(JOINT_DIR, "snap_joint_render.png")
    try:
        png = scene.save_image(resolution=[1280, 720])
        with open(out, 'wb') as f:
            f.write(png)
        print(f"[OK] Render guardado: {out}")
    except Exception as e:
        print(f"[WARN] No se pudo generar render: {e}")


def main():
    if not os.path.exists(URDF_PATH):
        print(f"[ERROR] No se encontró {URDF_PATH}")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if mode == "screenshot":
        export_screenshot()
        return

    # Intentar yourdfpy primero, luego trimesh
    try:
        view_with_yourdfpy()
    except Exception as e:
        print(f"[WARN] yourdfpy falló: {e}")
        print("[INFO] Usando trimesh como fallback...")
        try:
            view_with_trimesh()
        except Exception as e2:
            print(f"[WARN] Visor interactivo no disponible: {e2}")
            print("[INFO] Intentando render estático...")
            export_screenshot()


if __name__ == "__main__":
    main()
