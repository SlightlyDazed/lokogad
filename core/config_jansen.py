"""
config_jansen.py — Configuración del pipeline GAD para Theo Jansen.

Define: MN ranges, topología del grafo, carga de referencias,
función de fitness (comparación de trayectorias).
"""
import os
import math
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repositorio (core/ está un nivel abajo)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MESH_DIR = os.path.join(RESULT_DIR, "meshes")

# Referencia de trayectorias (tensores .pt)
GAIT_DIR = os.path.join(BASE_DIR, "opensim_extraction")
REF_FOOT_PATH = os.path.join(GAIT_DIR, "foot_ref_tensor.pt")
REF_KNEE_PATH = os.path.join(GAIT_DIR, "knee_ref_tensor.pt")
REF_HIP_PATH  = os.path.join(GAIT_DIR, "hip_ref_tensor.pt")

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR, MESH_DIR]:
    os.makedirs(d, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAGIC NUMBERS — ORIGINAL + BOUNDS
# ═══════════════════════════════════════════════════════════════════════════════
MN_ORIGINAL = np.array([38.0, 7.8, 15.0, 50.0, 41.5, 39.3, 61.9, 55.8,
                         40.1, 39.4, 36.7, 65.7, 49.0])
N_MN = len(MN_ORIGINAL)  # 13

# Factor de escala estimado: humano ~1030mm stride / jansen ~68mm ≈ 15x
SCALE_FACTOR = 15.0

# Bounds: permitimos variación amplia para que el optimizador escale
MN_LOWER = MN_ORIGINAL * 0.3   # mínimo: 30% del original
MN_UPPER = MN_ORIGINAL * 40.0  # máximo: 40x (para alcanzar escala humana ~15x)

# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGÍA DEL MECANISMO
# ═══════════════════════════════════════════════════════════════════════════════
PIVOT_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
N_PIVOTS = len(PIVOT_NAMES)
PIVOT_IDX = {name: i for i, name in enumerate(PIVOT_NAMES)}

# Barras: (nombre, nodo_inicio, nodo_fin, índice_MN)
# MN[0]=AB, MN[1]=ABy, MN[2]=AC, MN[3]=CD, MN[4]=BD, MN[5]=BE,
# MN[6]=CE, MN[7]=DF, MN[8]=BF, MN[9]=FG, MN[10]=EG, MN[11]=GH, MN[12]=EH
BAR_DEFS = [
    ("chassis",  'A', 'B', -1),   # longitud = sqrt(MN[0]²+MN[1]²)
    ("crank",    'A', 'C',  2),
    ("link_CD",  'C', 'D',  3),
    ("link_BD",  'B', 'D',  4),
    ("link_BE",  'B', 'E',  5),
    ("link_CE",  'C', 'E',  6),
    ("link_DF",  'D', 'F',  7),
    ("link_BF",  'B', 'F',  8),
    ("link_FG",  'F', 'G',  9),
    ("link_EG",  'E', 'G', 10),
    ("link_GH",  'G', 'H', 11),
    ("link_EH",  'E', 'H', 12),
]
N_BARS = len(BAR_DEFS)

# Mapeo punto → articulación humana
POINT_FOOT = 'H'
POINT_KNEE = 'G'
POINT_HIP  = 'A'

# ═══════════════════════════════════════════════════════════════════════════════
# CINEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

def linkage(p1, l1, p2, l2):
    """Intersección de dos círculos (CCW selection)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.sqrt(dx**2 + dy**2)
    if d > l1 + l2 or d < abs(l1 - l2) or d == 0:
        return None
    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h = math.sqrt(max(0, l1**2 - a**2))
    mx, my = p1[0] + dx * a / d, p1[1] + dy * a / d
    rx, ry = -dy * h / d, dx * h / d
    r1, r2 = (mx + rx, my + ry), (mx - rx, my - ry)
    cross = dx * (r1[1] - p1[1]) - dy * (r1[0] - p1[0])
    return r1 if cross < 0 else r2


def solve_mechanism(mn, angle_deg):
    """Resuelve cinemática para un conjunto de MN y un ángulo del crank."""
    A = (0.0, 0.0)
    B = (-mn[0], -mn[1])
    rad = math.radians(angle_deg)
    C = (mn[2] * math.cos(rad), mn[2] * math.sin(rad))
    D = linkage(C, mn[3], B, mn[4])
    E = linkage(B, mn[5], C, mn[6])
    F = linkage(D, mn[7], B, mn[8]) if D else None
    G = linkage(F, mn[9], E, mn[10]) if (F and E) else None
    H = linkage(G, mn[11], E, mn[12]) if (G and E) else None
    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H}


def compute_trajectories(mn, n_angles=360):
    """Computa trayectorias de todos los puntos para un ciclo completo.

    Returns:
        dict[str, np.array(n_angles, 2)] o None si el mecanismo falla.
    """
    trajectories = {name: [] for name in PIVOT_NAMES}
    for i in range(n_angles):
        angle = i * 360.0 / n_angles
        pts = solve_mechanism(mn, angle)
        # Verificar que todos los puntos son válidos
        for name in PIVOT_NAMES:
            if pts[name] is None:
                return None
            trajectories[name].append(pts[name])
    return {name: np.array(trajectories[name]) for name in PIVOT_NAMES}


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA DE REFERENCIAS
# ═══════════════════════════════════════════════════════════════════════════════

def load_reference_trajectories():
    """Carga los tensores de referencia y los convierte al plano sagital (X, Z) en mm.

    OpenSim: X=adelante, Y=lateral, Z=arriba.
    Mecanismo: x=horizontal, y=vertical.
    """
    foot_3d = torch.load(REF_FOOT_PATH, weights_only=True).cpu().numpy()  # (360, 3)
    knee_3d = torch.load(REF_KNEE_PATH, weights_only=True).cpu().numpy()
    hip_3d  = torch.load(REF_HIP_PATH,  weights_only=True).cpu().numpy()

    # Extraer plano sagital (X, Z) y convertir m → mm
    # Hacer relativo a la cadera
    hip_xz = np.column_stack([hip_3d[:, 0], hip_3d[:, 2]]) * 1000  # mm

    foot_xz = np.column_stack([foot_3d[:, 0], foot_3d[:, 2]]) * 1000 - hip_xz
    knee_xz = np.column_stack([knee_3d[:, 0], knee_3d[:, 2]]) * 1000 - hip_xz
    hip_rel = hip_xz - hip_xz  # (0, 0) siempre — la oscilación es el Z residual

    # Oscilación vertical de la cadera (target para el eje A fijo = mínima)
    hip_z_range = hip_3d[:, 2].max() - hip_3d[:, 2].min()  # en metros

    return {
        'foot': foot_xz,   # (360, 2) en mm, relativo a cadera
        'knee': knee_xz,   # (360, 2) en mm, relativo a cadera
        'hip_z_range_mm': hip_z_range * 1000,  # ~40mm
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS: PROCRUSTES ALIGNMENT — comparación en mm absolutos
# ═══════════════════════════════════════════════════════════════════════════════

def _procrustes_rotation(source, target):
    """Optimal rotation matrix (2×2) via SVD to align source → target.
    Both inputs must be centered (zero mean).
    Returns: rotation matrix R such that source @ R ≈ target.
    """
    H = source.T @ target  # (2, 2)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, d])  # Fix reflection
    R = (U @ D @ Vt).T  # transpose to get source→target
    # Corrección: R = V D U^T
    R = Vt.T @ D @ U.T
    return R


def _trajectory_error(mech_traj, ref_traj):
    """Compara dos trayectorias (N, 2) con Procrustes alignment.

    Procedimiento:
    1. Centrar ambas al centroide (eliminar traslación)
    2. Escalar mecanismo para match de tamaño con referencia
    3. Probar fase × reflexión → rotación óptima SVD → RMSD
    4. Retornar error en mm reales / span de referencia

    Returns:
        error: float (0 = perfecto match, 1+ = malo)
    """
    N = len(ref_traj)
    if len(mech_traj) != N:
        idx = np.linspace(0, len(mech_traj) - 1, N).astype(int)
        mech_traj = mech_traj[idx]

    # Centrar
    ref_c = ref_traj - ref_traj.mean(axis=0)
    mech_c = mech_traj - mech_traj.mean(axis=0)

    # Escalas RMS
    ref_scale = np.sqrt((ref_c ** 2).sum(axis=1).mean())
    mech_scale = np.sqrt((mech_c ** 2).sum(axis=1).mean())
    if ref_scale < 1e-6 or mech_scale < 1e-6:
        return 5.0

    # Error de escala (log-ratio)
    scale_ratio = mech_scale / ref_scale
    scale_error = abs(math.log(max(scale_ratio, 0.01)))

    # Escalar mecanismo para que coincida con escala de referencia
    mech_scaled = mech_c * (ref_scale / mech_scale)

    # Span de la referencia (para normalizar RMSD a valor relativo)
    ref_span = max(ref_c[:, 0].max() - ref_c[:, 0].min(),
                   ref_c[:, 1].max() - ref_c[:, 1].min())
    if ref_span < 1e-6:
        return 5.0

    # Buscar mejor: fase × reflexión → Procrustes
    best_rmsd = float('inf')
    coarse_step = max(1, N // 72)  # cada 5°

    for flip in [False, True]:
        ms = mech_scaled.copy()
        if flip:
            ms[:, 0] = -ms[:, 0]  # reflejar en X

        # Fase coarse
        best_offset_c = 0
        best_mse_c = float('inf')
        for offset in range(0, N, coarse_step):
            rolled = np.roll(ms, offset, axis=0)
            R = _procrustes_rotation(rolled, ref_c)
            aligned = rolled @ R
            mse = np.mean(np.sum((aligned - ref_c) ** 2, axis=1))
            if mse < best_mse_c:
                best_mse_c = mse
                best_offset_c = offset

        # Fase fina alrededor del mejor
        for offset in range(max(0, best_offset_c - coarse_step),
                            min(N, best_offset_c + coarse_step + 1)):
            rolled = np.roll(ms, offset, axis=0)
            R = _procrustes_rotation(rolled, ref_c)
            aligned = rolled @ R
            rmsd = math.sqrt(np.mean(np.sum((aligned - ref_c) ** 2, axis=1)))
            if rmsd < best_rmsd:
                best_rmsd = rmsd

    # Normalizar por span → error relativo
    shape_error = best_rmsd / ref_span
    total_error = shape_error + 0.3 * scale_error

    return total_error


def _position_error(mech_traj, ref_traj):
    """Error de posición: qué tan lejos está el centroide del mecanismo
    del centroide de la referencia, en mm.

    Ambas trayectorias son relativas a A/cadera.
    Normalizado por la distancia esperada (centroide ref).
    """
    ref_centroid = ref_traj.mean(axis=0)   # ej: (-168, -1020)
    mech_centroid = mech_traj.mean(axis=0) # ej: (-2274, -1728)

    dist_mm = np.sqrt(np.sum((mech_centroid - ref_centroid) ** 2))
    ref_dist = np.sqrt(np.sum(ref_centroid ** 2))

    if ref_dist < 1e-6:
        return dist_mm / 1000.0

    return dist_mm / ref_dist  # 0 = perfecto, 1 = error del 100%


def evaluate_mechanism(mn, ref_data):
    """Evalúa un conjunto de MN contra las trayectorias de referencia.

    Usa Procrustes alignment + error de posición absoluta.

    Args:
        mn: np.array(13,) — los magic numbers
        ref_data: dict del load_reference_trajectories()

    Returns:
        dict con: foot_error, knee_error, pos_error, stride_mm, ground_ratio, valid
    """
    trajs = compute_trajectories(mn)

    if trajs is None:
        return {
            'foot_error': 10.0, 'knee_error': 10.0, 'hip_error': 10.0,
            'pos_error': 10.0,
            'stride_mm': 0.0, 'ground_ratio': 0.0, 'valid': False
        }

    # Trayectorias del mecanismo (relativas a A = cadera)
    foot_mech = trajs[POINT_FOOT]  # H, (360, 2) en mm
    knee_mech = trajs[POINT_KNEE]  # G, (360, 2) en mm

    # Métricas de la marcha
    foot_x_range = foot_mech[:, 0].max() - foot_mech[:, 0].min()
    foot_y_range = foot_mech[:, 1].max() - foot_mech[:, 1].min()
    stride_mm = foot_x_range

    # Ground contact ratio
    y_min = foot_mech[:, 1].min()
    ground_thresh = y_min + 0.05 * foot_y_range
    ground_contact = np.sum(foot_mech[:, 1] < ground_thresh) / 360.0

    # Errores de trayectoria (Procrustes)
    foot_err = _trajectory_error(foot_mech, ref_data['foot'])
    knee_err = _trajectory_error(knee_mech, ref_data['knee'])

    # Error de posición absoluta (pie y rodilla deben estar en la posición correcta
    # relativa a A=cadera)
    foot_pos_err = _position_error(foot_mech, ref_data['foot'])
    knee_pos_err = _position_error(knee_mech, ref_data['knee'])
    pos_error = 0.6 * foot_pos_err + 0.4 * knee_pos_err

    hip_err = 0.0

    return {
        'foot_error': foot_err,
        'knee_error': knee_err,
        'hip_error': hip_err,
        'pos_error': pos_error,
        'stride_mm': stride_mm,
        'ground_ratio': ground_contact,
        'valid': True
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN MN ↔ [0, 1]
# ═══════════════════════════════════════════════════════════════════════════════

def mn_to_unit(mn):
    """Convierte MN a [0, 1] para el optimizador."""
    return (mn - MN_LOWER) / (MN_UPPER - MN_LOWER)

def unit_to_mn(x):
    """Convierte [0, 1] a MN reales."""
    return MN_LOWER + x * (MN_UPPER - MN_LOWER)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFO PyG
# ═══════════════════════════════════════════════════════════════════════════════

def mechanism_to_pyg(mn, targets=None):
    """Convierte un mecanismo a grafo PyG.

    Nodos (8): features = [0]*5 placeholder (se llenan con posiciones en runtime)
    Edges (24): bidireccional, features = [longitud_norm]
    Global features (13): MN normalizados

    Args:
        mn: np.array(13,) — magic numbers
        targets: np.array(5,) — [foot_err, knee_err, hip_err, stride, ground_ratio]

    Returns:
        torch_geometric.data.Data
    """
    from torch_geometric.data import Data

    # Node features: [is_fixed, is_crank, is_foot, is_knee, role_encoding]
    node_feats = np.zeros((N_PIVOTS, 5), dtype=np.float32)
    node_feats[PIVOT_IDX['A'], 0] = 1.0  # fixed
    node_feats[PIVOT_IDX['B'], 0] = 1.0  # fixed
    node_feats[PIVOT_IDX['C'], 1] = 1.0  # crank endpoint
    node_feats[PIVOT_IDX['H'], 2] = 1.0  # foot
    node_feats[PIVOT_IDX['G'], 3] = 1.0  # knee analog
    # role encoding: positional encoding by topology depth
    depths = {'A': 0, 'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 3, 'G': 4, 'H': 5}
    for name, depth in depths.items():
        node_feats[PIVOT_IDX[name], 4] = depth / 5.0

    # Edge construction (bidirectional)
    edge_src, edge_dst, edge_attr = [], [], []
    for _, p1_name, p2_name, mn_idx in BAR_DEFS:
        i, j = PIVOT_IDX[p1_name], PIVOT_IDX[p2_name]
        if mn_idx == -1:
            length = math.sqrt(mn[0]**2 + mn[1]**2)
        else:
            length = mn[mn_idx]

        # Normalizar longitud al rango del MN
        length_norm = length / 1000.0  # a metros para escala uniforme

        # Bidireccional
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_attr.extend([[length_norm], [length_norm]])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Global features: MN normalizados a [0, 1]
    global_feats = torch.tensor(mn_to_unit(mn), dtype=torch.float)

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_feats=global_feats,
    )

    if targets is not None:
        data.y = torch.tensor(targets, dtype=torch.float)

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# GNN CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
N_NODE_FEATS = 5
N_EDGE_FEATS = 1
N_GLOBAL_FEATS = N_MN  # 13
N_TARGETS = 5  # foot_err, knee_err, hip_err, stride_mm, ground_ratio
N_ENSEMBLE = 3
MAX_EPOCHS = 1000
PATIENCE = 60
LR = 1e-3
BATCH_SIZE = 512
DROPOUT = 0.1

# NSGA
N_OBJ = 4     # foot_error, knee_error, ground_ratio_error, pos_error
N_CONSTR = 2  # valid mechanism, stride > 0
POP_SIZE = 10000
N_GEN = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
