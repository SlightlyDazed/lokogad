"""
general_solver.py — Solver cinemático para linkages planos arbitrarios.

Resuelve la posición de todos los pivotes dado:
- Grafo de conectividad (nodos + barras)
- Posiciones fijas (ground, crank)
- Ángulo del crank

Método: propagación BFS desde nodos conocidos.
Cuando un nodo libre tiene ≥2 vecinos con posición conocida → intersección de círculos.
"""
import math
import numpy as np
from collections import deque


def circle_intersection(p1, r1, p2, r2, prefer_ccw=True):
    """Intersección de dos círculos. Retorna un punto o None.

    Args:
        p1, p2: centros (x, y)
        r1, r2: radios
        prefer_ccw: si True, selecciona el punto CCW respecto a p1→p2

    Returns:
        (x, y) o None si no intersectan
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    d = math.sqrt(dx * dx + dy * dy)

    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-10:
        return None

    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h_sq = r1 * r1 - a * a
    if h_sq < 0:
        h_sq = 0
    h = math.sqrt(h_sq)

    mx = p1[0] + dx * a / d
    my = p1[1] + dy * a / d

    rx = -dy * h / d
    ry = dx * h / d

    sol1 = (mx + rx, my + ry)
    sol2 = (mx - rx, my - ry)

    # CCW test: cross product of (p1→p2) × (p1→sol)
    cross = dx * (sol1[1] - p1[1]) - dy * (sol1[0] - p1[0])

    if prefer_ccw:
        return sol1 if cross < 0 else sol2
    else:
        return sol1 if cross >= 0 else sol2


def solve_linkage(nodes, edges, crank_angle_deg, ground_pos=None):
    """Resuelve la cinemática de un linkage plano arbitrario.

    Args:
        nodes: list of dict con keys:
            'id': int
            'type': 'ground' | 'crank' | 'free' | 'output_foot' | 'output_knee'
            'x': float (offset x para ground nodes)
            'y': float (offset y para ground nodes)
            'crank_radius': float (solo para type='crank')
        edges: list of dict con keys:
            'n1': int (node id)
            'n2': int (node id)
            'length': float
        crank_angle_deg: ángulo del crank en grados
        ground_pos: (x, y) posición del ground principal. Default (0, 0).

    Returns:
        dict[int, (float, float)] — posiciones de todos los nodos, o None si falla
    """
    if ground_pos is None:
        ground_pos = (0.0, 0.0)

    # Build adjacency
    n_nodes = len(nodes)
    node_map = {n['id']: n for n in nodes}
    adj = {n['id']: [] for n in nodes}
    for e in edges:
        adj[e['n1']].append((e['n2'], e['length']))
        adj[e['n2']].append((e['n1'], e['length']))

    # Posiciones conocidas
    positions = {}
    queue = deque()

    for n in nodes:
        nid = n['id']
        if n['type'] == 'ground':
            positions[nid] = (ground_pos[0] + n.get('x', 0),
                              ground_pos[1] + n.get('y', 0))
            queue.append(nid)
        elif n['type'] == 'crank':
            # El crank rota alrededor de su ground parent
            r = n.get('crank_radius', 0)
            # Buscar el ground al que está conectado
            parent_pos = ground_pos
            for neighbor_id, _ in adj[nid]:
                if node_map[neighbor_id]['type'] == 'ground':
                    if neighbor_id in positions:
                        parent_pos = positions[neighbor_id]
                    break
            rad = math.radians(crank_angle_deg)
            positions[nid] = (parent_pos[0] + r * math.cos(rad),
                              parent_pos[1] + r * math.sin(rad))
            queue.append(nid)

    # BFS: propagar posiciones
    max_iterations = n_nodes * 3  # prevent infinite loop
    iteration = 0

    while len(positions) < n_nodes and iteration < max_iterations:
        iteration += 1

        # Intentar resolver cada nodo que aún no tiene posición
        progress = False
        for n in nodes:
            nid = n['id']
            if nid in positions:
                continue

            # Buscar vecinos con posición conocida
            known_neighbors = []
            for neighbor_id, bar_length in adj[nid]:
                if neighbor_id in positions:
                    known_neighbors.append((neighbor_id, bar_length))

            if len(known_neighbors) >= 2:
                # Tenemos 2+ puntos conocidos → resolver por intersección
                n1_id, l1 = known_neighbors[0]
                n2_id, l2 = known_neighbors[1]
                p1 = positions[n1_id]
                p2 = positions[n2_id]

                result = circle_intersection(p1, l1, p2, l2, prefer_ccw=True)
                if result is not None:
                    positions[nid] = result
                    progress = True
                else:
                    # Intentar con el otro par si hay más de 2
                    for i in range(2, len(known_neighbors)):
                        n2_id, l2 = known_neighbors[i]
                        p2 = positions[n2_id]
                        result = circle_intersection(p1, l1, p2, l2, prefer_ccw=True)
                        if result is not None:
                            positions[nid] = result
                            progress = True
                            break

        if not progress:
            # No se resolvió ningún nodo → mecanismo no ensambla o topología inválida
            break

    if len(positions) < n_nodes:
        return None

    return positions


def compute_full_trajectory(nodes, edges, n_angles=360):
    """Computa la trayectoria de todos los nodos para un ciclo completo.

    Returns:
        dict[int, np.array(n_angles, 2)] o None si falla en algún ángulo
    """
    all_trajectories = {n['id']: [] for n in nodes}

    for i in range(n_angles):
        angle = i * 360.0 / n_angles
        positions = solve_linkage(nodes, edges, angle)
        if positions is None:
            return None
        for nid, pos in positions.items():
            all_trajectories[nid].append(pos)

    return {nid: np.array(pts) for nid, pts in all_trajectories.items()}


def find_output_nodes(nodes, trajectories):
    """Identifica automáticamente pie y rodilla basándose en la posición.

    - Pie = nodo con el centroide Y más bajo (más negativo)
    - Rodilla = nodo a ~medio camino vertical entre cadera y pie

    Returns:
        (foot_id, knee_id)
    """
    # Excluir ground y crank
    free_nodes = [n for n in nodes
                  if n['type'] in ('free', 'output_foot', 'output_knee')]

    if not free_nodes:
        return None, None

    # Centroide Y de cada nodo
    centroids = {}
    for n in free_nodes:
        traj = trajectories.get(n['id'])
        if traj is not None:
            centroids[n['id']] = traj[:, 1].mean()

    if not centroids:
        return None, None

    # Pie = menor Y (más abajo)
    foot_id = min(centroids, key=centroids.get)

    # Rodilla = el que esté más cerca del punto medio vertical
    ground_y = 0.0  # A está en origen
    foot_y = centroids[foot_id]
    mid_y = (ground_y + foot_y) / 2

    remaining = {k: v for k, v in centroids.items() if k != foot_id}
    if remaining:
        knee_id = min(remaining, key=lambda k: abs(remaining[k] - mid_y))
    else:
        knee_id = None

    return foot_id, knee_id
