#!/usr/bin/env python3
"""
view_urdf_animated.py — Animación del snap pin joint
Muestra link_A estático y link_B oscilando ±90° en el eje Z.
Ejecutar: source ~/pytorch-rocm/bin/activate && python3 view_urdf_animated.py
"""
import os
import sys
import math
import time
import numpy as np
import trimesh
import trimesh.viewer
from trimesh.viewer.windowed import SceneViewer

JOINT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parámetros del joint (en metros, convertido de mm)
JX = 0.054   # posición X del pin
JZ = 0.004   # posición Z del pin (top de link_A body)
SPEED = 1.5  # velocidad de oscilación (rad/s)
AMPLITUDE = math.radians(90)  # ±90°

# Estado global de animación
_start_time = None
_mesh_b_base = None


def load_scene():
    """Carga la escena con ambos links."""
    global _mesh_b_base

    scene = trimesh.Scene()

    # Link A — estático, color gris claro
    mesh_a = trimesh.load(os.path.join(JOINT_DIR, "link_A.stl"))
    mesh_a.visual.face_colors = [200, 200, 210, 255]
    mesh_a.apply_scale(0.001)
    scene.add_geometry(mesh_a, node_name="link_A", geom_name="link_A")

    # Link B — se animará, color azul
    mesh_b = trimesh.load(os.path.join(JOINT_DIR, "link_B.stl"))
    mesh_b.visual.face_colors = [60, 120, 220, 255]
    mesh_b.apply_scale(0.001)

    # Guardar mesh original para recalcular transforms
    _mesh_b_base = mesh_b.copy()

    # Posición inicial: encima de link_A en el punto del joint
    T_init = np.eye(4)
    T_init[2, 3] = JZ
    mesh_b.apply_transform(T_init)
    scene.add_geometry(mesh_b, node_name="link_B", geom_name="link_B")

    # Eje de referencia para visualizar el joint
    axis = trimesh.creation.cylinder(radius=0.0005, height=0.015,
                                      sections=12)
    axis.visual.face_colors = [255, 50, 50, 200]
    T_axis = np.eye(4)
    T_axis[0, 3] = JX
    T_axis[2, 3] = JZ + 0.005
    axis.apply_transform(T_axis)
    scene.add_geometry(axis, node_name="joint_axis", geom_name="joint_axis")

    return scene


def animate_callback(scene):
    """Callback llamado cada frame para animar link_B."""
    global _start_time

    if _start_time is None:
        _start_time = time.time()

    t = time.time() - _start_time
    angle = AMPLITUDE * math.sin(SPEED * t)

    # Construir transformación: trasladar a posición + rotar alrededor del pin
    T_offset = np.eye(4)
    T_offset[2, 3] = JZ  # elevar encima de link_A

    R = trimesh.transformations.rotation_matrix(
        angle, [0, 0, 1], point=[JX, 0, 0]
    )

    final_transform = R @ T_offset

    # Actualizar transformación en el grafo de la escena
    scene.graph.update("link_B", matrix=final_transform)


def main():
    print("=" * 60)
    print("  Snap Pin Joint — Animación interactiva")
    print("=" * 60)
    print(f"  Joint:     revolute, eje Z")
    print(f"  Amplitud:  ±{math.degrees(AMPLITUDE):.0f}°")
    print(f"  Pin en:    x={JX*1000:.0f}mm, z={JZ*1000:.0f}mm")
    print(f"  Controles: arrastra para rotar vista, scroll para zoom")
    print(f"  Cerrar:    cierra la ventana")
    print("=" * 60)

    scene = load_scene()

    print(f"\n[OK] Escena cargada: {len(scene.geometry)} objetos")
    for name, geom in scene.geometry.items():
        print(f"     {name}: {len(geom.faces)} caras")

    print("\n[INFO] Abriendo visor animado...")

    # Abrir visor con callback de animación
    SceneViewer(
        scene,
        callback=animate_callback,
        callback_period=1.0/30.0,  # 30 FPS
        resolution=(1280, 720),
        window_title="GAD Snap Pin Joint — Animación ±90°",
    )


if __name__ == "__main__":
    main()
