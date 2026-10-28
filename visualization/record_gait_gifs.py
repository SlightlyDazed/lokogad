#!/usr/bin/env python3
"""
record_gait_gifs.py — Graba escenas de 5 segundos en GIF del mecanismo evolucionado
en 4 perspectivas clave:
  1. Lateral (Plano XY) — Comparativa con trayectoria humana
  2. Frontal — Muestra las capas Z separadas
  3. 3/4 Isométrica — Vista 3D vistosa para README
  4. Órbita 360° — Cámara girando suavemente alrededor del mecanismo
"""
import os, sys, json, math, time, subprocess, shutil, tempfile
import numpy as np
import trimesh, io, pyglet
from trimesh.viewer.windowed import SceneViewer
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from general_solver import solve_linkage, compute_full_trajectory
from linkage_genome import LinkageGenome
from config_jansen import load_reference_trajectories, RESULT_DIR

# ── Configuración de Render y GIF ─────────────────────────────────────────────
DURATION_SEC = 5.0
FPS = 20
TOTAL_FRAMES = int(DURATION_SEC * FPS)  # 100 frames
CYCLES = 2  # 2 ciclos completos en 5s -> loop perfecto sin saltos
RESOLUTION = (800, 500)

GIF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs")
os.makedirs(GIF_DIR, exist_ok=True)

# ── Dimensiones (piezas gruesas para alta visibilidad) ────────────────────────
BAR_WIDTH = 25.0   # mm
BAR_THICK = 10.0   # mm
PIN_R = 7.5        # mm
LAYER_PITCH = 16.0 # mm

COLORS = [
    [255, 70, 70, 255], [70, 200, 70, 255], [70, 70, 220, 255],
    [220, 200, 50, 255], [200, 70, 200, 255], [70, 200, 200, 255],
    [140, 140, 210, 255], [220, 140, 50, 255], [50, 140, 220, 255],
    [230, 50, 130, 255], [50, 230, 130, 255], [180, 100, 60, 255],
]

def compute_z_layers(edges):
    n = len(edges)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if {edges[i]['n1'], edges[i]['n2']} & {edges[j]['n1'], edges[j]['n2']}:
                conflicts[i].add(j)
                conflicts[j].add(i)
    layers = [None] * n
    for i in range(n):
        used = {layers[j] for j in conflicts[i] if layers[j] is not None}
        for k in range(n + 1):
            cand = (k + 1) // 2 * (1 if k % 2 == 0 else -1)
            if cand not in used:
                layers[i] = cand
                break
    return layers

def build_scene():
    genome_path = os.path.join(RESULT_DIR, "best_genome.json")
    with open(genome_path) as f:
        data = json.load(f)
    g = LinkageGenome.from_dict(data)
    ref = load_reference_trajectories()

    trajs = compute_full_trajectory(g.nodes, g.edges)
    if trajs is None:
        raise RuntimeError("El mecanismo no ensambla")

    foot_id = g.foot_id or 7
    knee_id = g.knee_id or 6
    foot_traj = trajs[foot_id]

    mech_flipped = np.column_stack([-foot_traj[:, 0], foot_traj[:, 1]])
    offset = ref['foot'].mean(axis=0) - mech_flipped.mean(axis=0)
    OX, OY = offset[0] / 1000, offset[1] / 1000

    z_layers = compute_z_layers(g.edges)
    node_bars = {n['id']: [] for n in g.nodes}
    for i, e in enumerate(g.edges):
        node_bars[e['n1']].append(i)
        node_bars[e['n2']].append(i)

    scene = trimesh.Scene()
    bar_meshes = {}
    bar_z = {}

    for i, e in enumerate(g.edges):
        L = e['length']
        hw = BAR_WIDTH / 2
        z_mm = z_layers[i] * LAYER_PITCH

        body = trimesh.creation.box(extents=[L, BAR_WIDTH, BAR_THICK])
        body.apply_translation([L / 2, 0, 0])
        cap1 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=16)
        cap2 = trimesh.creation.cylinder(radius=hw, height=BAR_THICK, sections=16)
        cap2.apply_translation([L, 0, 0])
        bar = trimesh.util.concatenate([body, cap1, cap2])
        bar.visual.face_colors = COLORS[i % len(COLORS)]
        bar.apply_scale(0.001)

        name = f"bar_{i}"
        bar_meshes[name] = (e['n1'], e['n2'])
        bar_z[name] = z_mm / 1000
        scene.add_geometry(bar.copy(), node_name=name, geom_name=name)

    pin_z_center = {}
    for n in g.nodes:
        nid = n['id']
        connected = node_bars[nid]
        if connected:
            z_vals = [z_layers[bi] * LAYER_PITCH for bi in connected]
            zmin = min(z_vals) - BAR_THICK / 2
            zmax = max(z_vals) + BAR_THICK / 2
            height = max(zmax - zmin, BAR_THICK * 2)
        else:
            zmin, zmax, height = -BAR_THICK, BAR_THICK, BAR_THICK * 2

        center_z = (zmin + zmax) / 2
        pin_z_center[nid] = center_z / 1000

        pin = trimesh.creation.cylinder(
            radius=PIN_R * 0.001, height=height * 0.001, sections=16
        )
        if n['type'] == 'ground':
            pin.visual.face_colors = [255, 255, 255, 230]
        elif n['type'] == 'crank':
            pin.visual.face_colors = [255, 50, 50, 230]
        elif n['type'] == 'output_foot':
            pin.visual.face_colors = [50, 150, 255, 230]
        elif n['type'] == 'output_knee':
            pin.visual.face_colors = [50, 255, 150, 230]
        else:
            pin.visual.face_colors = [255, 220, 50, 230]

        scene.add_geometry(pin, node_name=f"pin_{nid}", geom_name=f"pin_{nid}")

    # Trazas de referencia
    fx = -foot_traj[:, 0] / 1000 + OX
    fy = foot_traj[:, 1] / 1000 + OY
    pts3d = np.column_stack([fx, fy, np.full(len(fx), -0.015)])
    fp = trimesh.load_path(pts3d)
    fp.colors = [[230, 50, 50, 200]] * len(fp.entities)
    scene.add_geometry(fp, node_name="foot_trace", geom_name="foot_trace")

    rf = ref['foot'] / 1000
    rf3d = np.column_stack([rf[:, 0], rf[:, 1], np.full(len(rf), -0.018)])
    rp = trimesh.load_path(rf3d)
    rp.colors = [[50, 150, 255, 200]] * len(rp.entities)
    scene.add_geometry(rp, node_name="ref_foot", geom_name="ref_foot")

    rk = ref['knee'] / 1000
    rk3d = np.column_stack([rk[:, 0], rk[:, 1], np.full(len(rk), -0.018)])
    rkp = trimesh.load_path(rk3d)
    rkp.colors = [[50, 255, 150, 200]] * len(rkp.entities)
    scene.add_geometry(rkp, node_name="ref_knee", geom_name="ref_knee")

    # Suelo
    ground_y = min(min(fy), rf[:, 1].min()) - 0.01
    ground = trimesh.creation.box(extents=[3.0, 0.003, 0.8])
    ground.visual.face_colors = [40, 180, 40, 50]
    Tg = np.eye(4)
    Tg[1, 3] = ground_y
    ground.apply_transform(Tg)
    scene.add_geometry(ground, node_name="gnd", geom_name="gnd")

    return g, scene, bar_meshes, bar_z, pin_z_center, OX, OY

def update_kinematics(g, scene, bar_meshes, bar_z, pin_z_center, OX, OY, angle_deg):
    positions = solve_linkage(g.nodes, g.edges, angle_deg)
    if positions is None:
        return False

    for name, (n1, n2) in bar_meshes.items():
        if n1 in positions and n2 in positions:
            p1, p2 = positions[n1], positions[n2]
            sx, sy = -p1[0] / 1000 + OX, p1[1] / 1000 + OY
            ex, ey = -p2[0] / 1000 + OX, p2[1] / 1000 + OY
            a = math.atan2(ey - sy, ex - sx)
            T = np.eye(4)
            T[0, 0], T[0, 1] = math.cos(a), -math.sin(a)
            T[1, 0], T[1, 1] = math.sin(a), math.cos(a)
            T[0, 3], T[1, 3], T[2, 3] = sx, sy, bar_z[name]
            scene.graph.update(name, matrix=T)

    for n in g.nodes:
        nid = n['id']
        if nid in positions:
            T = np.eye(4)
            T[0, 3] = -positions[nid][0] / 1000 + OX
            T[1, 3] = positions[nid][1] / 1000 + OY
            T[2, 3] = pin_z_center[nid]
            scene.graph.update(f"pin_{nid}", matrix=T)

    return True

def frames_to_gif(frame_dir, out_gif_path, fps=FPS):
    """Convierte frames PNG a GIF optimizado con paleta de alta calidad mediante ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-filter_complex",
        f"[0:v] fps={fps},split [a][b];[a] palettegen=stats_mode=diff [p];[b][p] paletteuse=dither=bayer:bayer_scale=3",
        out_gif_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def record_view(view_name, get_camera_params, out_filename):
    print(f"\n[+] Grabando vista: {view_name} (5s, {TOTAL_FRAMES} frames)...")
    g, scene, bar_meshes, bar_z, pin_z_center, OX, OY = build_scene()

    tmpdir = tempfile.mkdtemp(prefix=f"gait_{out_filename}_")
    try:
        # Inicializar ventana oculta reutilizable para alta velocidad
        window = SceneViewer(
            scene,
            start_loop=False,
            visible=False,
            resolution=RESOLUTION,
            fullscreen=False,
            resizable=False,
        )

        for frame_idx in range(TOTAL_FRAMES):
            t = frame_idx / TOTAL_FRAMES
            angle = (t * CYCLES * 360.0) % 360.0
            update_kinematics(g, scene, bar_meshes, bar_z, pin_z_center, OX, OY, angle)

            # Actualizar cámara
            angles, distance, center = get_camera_params(t)
            scene.set_camera(angles=angles, distance=distance, center=center)

            pyglet.clock.tick()
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event("on_draw")
            window.flip()

            bio = io.BytesIO()
            window.save_image(bio)
            bio.seek(0)
            img = Image.open(bio)
            img.save(os.path.join(tmpdir, f"frame_{frame_idx:04d}.png"))

            if (frame_idx + 1) % 25 == 0 or frame_idx == TOTAL_FRAMES - 1:
                print(f"    Frame {frame_idx + 1}/{TOTAL_FRAMES} ({int((frame_idx+1)/TOTAL_FRAMES*100)}%)")

        window.close()

        out_gif = os.path.join(GIF_DIR, f"{out_filename}.gif")
        print(f"    Generando GIF optimizado con ffmpeg...")
        frames_to_gif(tmpdir, out_gif, fps=FPS)
        size_mb = os.path.getsize(out_gif) / (1024 * 1024)
        print(f"    ✅ Guardado: {out_gif} ({size_mb:.2f} MB)")
        return out_gif
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def main():
    print("=" * 65)
    print("  RECORD GAIT GIFS — Mecanismo Evolucionado (4 Vistas)")
    print(f"  Duración: {DURATION_SEC}s | FPS: {FPS} | Resolución: {RESOLUTION}")
    print(f"  Destino: {GIF_DIR}")
    print("=" * 65)

    center = [-0.20, -0.42, 0.0]

    # 1. Lateral (Plano XY)
    record_view(
        view_name="Lateral (Plano XY)",
        get_camera_params=lambda t: ([0.0, 0.0, 0.0], 2.2, center),
        out_filename="gait_lateral_xy"
    )

    # 2. Frontal (Capas Z separadas)
    record_view(
        view_name="Frontal (Capas Z separadas)",
        get_camera_params=lambda t: ([math.radians(12), math.radians(76), 0.0], 2.1, center),
        out_filename="gait_frontal_zlayers"
    )

    # 3. 3/4 o Isométrica (Vista README)
    record_view(
        view_name="3/4 Isométrica (Vista README)",
        get_camera_params=lambda t: ([math.radians(18), math.radians(35), 0.0], 2.2, center),
        out_filename="gait_isometric_3d"
    )

    # 4. Órbita 360° (Giro lento alrededor del mecanismo)
    def orbit_params(t):
        yaw = 2 * math.pi * t  # 360 grados en 5 segundos
        pitch = math.radians(18)
        return ([pitch, yaw, 0.0], 2.3, center)

    record_view(
        view_name="Órbita 360° (Giro continuo)",
        get_camera_params=orbit_params,
        out_filename="gait_orbit_360"
    )

    print("\n" + "=" * 65)
    print("  TODAS LAS VISTAS FUERON GRABADAS CON ÉXITO")
    print(f"  Archivos generados en: {GIF_DIR}")
    for fname in ["gait_lateral_xy.gif", "gait_frontal_zlayers.gif", "gait_isometric_3d.gif", "gait_orbit_360.gif"]:
        path = os.path.join(GIF_DIR, fname)
        if os.path.exists(path):
            print(f"   • {fname} ({os.path.getsize(path)/(1024*1024):.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    main()
