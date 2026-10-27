"""
linkage_genome.py — Genoma de longitud variable para linkages planos.

Define un mecanismo como un grafo mutable (nodos=pivotes, edges=barras)
con operaciones de mutación estructural para evolución topológica.
"""
import math
import copy
import random
import numpy as np
from general_solver import solve_linkage, compute_full_trajectory, find_output_nodes


# ═══════════════════════════════════════════════════════════════════════════════
# GENOMA
# ═══════════════════════════════════════════════════════════════════════════════

class LinkageGenome:
    """Mecanismo como grafo mutable.

    Nodos: pivotes con tipo (ground, crank, free, output_foot, output_knee)
    Edges: barras con longitud
    """

    def __init__(self):
        self.nodes = []     # list of dicts
        self.edges = []     # list of dicts
        self._next_id = 0
        self.fitness = None
        self.foot_id = None
        self.knee_id = None

    def add_node(self, ntype, x=0.0, y=0.0, crank_radius=0.0):
        """Agrega un nodo y retorna su ID."""
        nid = self._next_id
        self._next_id += 1
        self.nodes.append({
            'id': nid,
            'type': ntype,
            'x': x,
            'y': y,
            'crank_radius': crank_radius,
        })
        return nid

    def add_edge(self, n1, n2, length):
        """Agrega una barra entre n1 y n2."""
        # No permitir duplicados
        for e in self.edges:
            if (e['n1'] == n1 and e['n2'] == n2) or \
               (e['n1'] == n2 and e['n2'] == n1):
                return
        self.edges.append({'n1': n1, 'n2': n2, 'length': length})

    def remove_edge(self, n1, n2):
        """Elimina la barra entre n1 y n2."""
        self.edges = [e for e in self.edges
                      if not ((e['n1'] == n1 and e['n2'] == n2) or
                              (e['n1'] == n2 and e['n2'] == n1))]

    def get_node(self, nid):
        for n in self.nodes:
            if n['id'] == nid:
                return n
        return None

    def neighbors(self, nid):
        """Retorna lista de (neighbor_id, bar_length)."""
        result = []
        for e in self.edges:
            if e['n1'] == nid:
                result.append((e['n2'], e['length']))
            elif e['n2'] == nid:
                result.append((e['n1'], e['length']))
        return result

    def is_connected(self):
        """Verifica que el grafo es conexo."""
        if not self.nodes:
            return False
        visited = set()
        queue = [self.nodes[0]['id']]
        node_ids = {n['id'] for n in self.nodes}
        while queue:
            nid = queue.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for nb, _ in self.neighbors(nid):
                if nb not in visited:
                    queue.append(nb)
        return visited == node_ids

    def node_degree(self, nid):
        return len(self.neighbors(nid))

    def n_free_nodes(self):
        return sum(1 for n in self.nodes if n['type'] in ('free', 'output_foot', 'output_knee'))

    def copy(self):
        g = LinkageGenome()
        g.nodes = copy.deepcopy(self.nodes)
        g.edges = copy.deepcopy(self.edges)
        g._next_id = self._next_id
        g.fitness = self.fitness
        g.foot_id = self.foot_id
        g.knee_id = self.knee_id
        return g

    def to_dict(self):
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'foot_id': self.foot_id,
            'knee_id': self.knee_id,
        }

    @staticmethod
    def from_dict(d):
        g = LinkageGenome()
        g.nodes = d['nodes']
        g.edges = d['edges']
        g._next_id = max(n['id'] for n in g.nodes) + 1 if g.nodes else 0
        g.foot_id = d.get('foot_id')
        g.knee_id = d.get('knee_id')
        return g

    def __repr__(self):
        return (f"LinkageGenome(nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"foot={self.foot_id}, knee={self.knee_id})")


# ═══════════════════════════════════════════════════════════════════════════════
# SEMILLA: CREAR GENOMA JANSEN
# ═══════════════════════════════════════════════════════════════════════════════

def create_jansen_genome(scale=1.0):
    """Crea el genoma del mecanismo de Jansen original a una escala dada."""
    MN = np.array([38.0, 7.8, 15.0, 50.0, 41.5, 39.3, 61.9,
                   55.8, 40.1, 39.4, 36.7, 65.7, 49.0]) * scale
    g = LinkageGenome()

    # Nodos
    id_A = g.add_node('ground', x=0.0, y=0.0)           # 0: A — cadera
    id_B = g.add_node('ground', x=-MN[0], y=-MN[1])      # 1: B — chassis
    id_C = g.add_node('crank', crank_radius=MN[2])        # 2: C — crank tip
    id_D = g.add_node('free')                              # 3: D
    id_E = g.add_node('free')                              # 4: E
    id_F = g.add_node('free')                              # 5: F
    id_G = g.add_node('output_knee')                       # 6: G — rodilla
    id_H = g.add_node('output_foot')                       # 7: H — pie

    # Barras (12)
    g.add_edge(id_A, id_C, MN[2])   # crank
    g.add_edge(id_C, id_D, MN[3])
    g.add_edge(id_B, id_D, MN[4])
    g.add_edge(id_B, id_E, MN[5])
    g.add_edge(id_C, id_E, MN[6])
    g.add_edge(id_D, id_F, MN[7])
    g.add_edge(id_B, id_F, MN[8])
    g.add_edge(id_F, id_G, MN[9])
    g.add_edge(id_E, id_G, MN[10])
    g.add_edge(id_G, id_H, MN[11])
    g.add_edge(id_E, id_H, MN[12])

    g.foot_id = id_H
    g.knee_id = id_G

    return g


def create_simple_4bar(scale=15.0):
    """Crea un 4-bar linkage simple como semilla alternativa."""
    g = LinkageGenome()
    L = 60.0 * scale  # ~900mm

    id_A = g.add_node('ground', x=0.0, y=0.0)
    id_B = g.add_node('ground', x=-L*0.4, y=-L*0.1)
    id_C = g.add_node('crank', crank_radius=L*0.25)
    id_D = g.add_node('free')
    id_F = g.add_node('output_foot')

    g.add_edge(id_A, id_C, L*0.25)
    g.add_edge(id_C, id_D, L*0.8)
    g.add_edge(id_B, id_D, L*0.7)
    g.add_edge(id_D, id_F, L*1.0)
    g.add_edge(id_B, id_F, L*0.9)

    g.foot_id = id_F
    g.knee_id = id_D
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# MUTACIONES
# ═══════════════════════════════════════════════════════════════════════════════

MAX_NODES = 16
MAX_EDGES = 30
MIN_EDGES = 4


def mutate(genome, rng=None):
    """Aplica una mutación al genoma. Retorna el genoma mutado (copia)."""
    if rng is None:
        rng = random.Random()

    g = genome.copy()
    g.fitness = None

    # Elegir tipo de mutación
    n_free = g.n_free_nodes()
    n_edges = len(g.edges)

    mutations = [
        ('mutate_length',  0.50),
        ('mutate_ground',  0.12),
        ('add_pivot',      0.08 if n_free < MAX_NODES - 3 else 0),
        ('add_bar',        0.06 if n_edges < MAX_EDGES else 0),
        ('remove_bar',     0.06 if n_edges > MIN_EDGES else 0),
        ('remove_pivot',   0.04 if n_free > 3 else 0),
        ('mutate_crank',   0.09),
        ('reassign_output', 0.05),
    ]

    total = sum(p for _, p in mutations)
    mutations = [(m, p/total) for m, p in mutations]

    r = rng.random()
    cumul = 0
    chosen = 'mutate_length'
    for name, prob in mutations:
        cumul += prob
        if r < cumul:
            chosen = name
            break

    if chosen == 'mutate_length':
        _mutate_length(g, rng)
    elif chosen == 'mutate_ground':
        _mutate_ground(g, rng)
    elif chosen == 'add_pivot':
        _add_pivot(g, rng)
    elif chosen == 'add_bar':
        _add_bar(g, rng)
    elif chosen == 'remove_bar':
        _remove_bar(g, rng)
    elif chosen == 'remove_pivot':
        _remove_pivot(g, rng)
    elif chosen == 'mutate_crank':
        _mutate_crank(g, rng)
    elif chosen == 'reassign_output':
        _reassign_output(g, rng)

    return g


def _mutate_length(g, rng):
    """Cambia la longitud de una barra ±30%."""
    if not g.edges:
        return
    e = rng.choice(g.edges)
    factor = 1.0 + rng.gauss(0, 0.15)
    factor = max(0.3, min(3.0, factor))
    e['length'] = max(10.0, e['length'] * factor)


def _mutate_ground(g, rng):
    """Mover un ground node ±20%."""
    grounds = [n for n in g.nodes if n['type'] == 'ground']
    if len(grounds) < 2:
        return
    # No mover A (el primero), solo B y otros
    moveable = [n for n in grounds if n['id'] != 0]
    if not moveable:
        return
    n = rng.choice(moveable)
    n['x'] *= (1.0 + rng.gauss(0, 0.15))
    n['y'] *= (1.0 + rng.gauss(0, 0.15))


def _mutate_crank(g, rng):
    """Modificar radio del crank ±25%."""
    cranks = [n for n in g.nodes if n['type'] == 'crank']
    if not cranks:
        return
    c = cranks[0]
    factor = 1.0 + rng.gauss(0, 0.12)
    c['crank_radius'] = max(10.0, c['crank_radius'] * factor)
    # También actualizar la barra crank correspondiente
    for e in g.edges:
        if (e['n1'] == 0 and e['n2'] == c['id']) or \
           (e['n2'] == 0 and e['n1'] == c['id']):
            e['length'] = c['crank_radius']
            break


def _add_pivot(g, rng):
    """Divide una barra insertando un pivote intermedio.

    Resuelve el mecanismo primero para usar posiciones reales al calcular
    longitudes geométricamente factibles.
    """
    if not g.edges:
        return

    # Resolver mecanismo para obtener posiciones reales
    from general_solver import solve_linkage
    positions = solve_linkage(g.nodes, g.edges, rng.uniform(0, 360))

    e = rng.choice(g.edges)
    n1, n2, L = e['n1'], e['n2'], e['length']

    # Split ratio
    ratio = rng.uniform(0.35, 0.65)
    L1, L2 = L * ratio, L * (1 - ratio)

    # Nuevo pivote
    new_id = g.add_node('free')

    # Reemplazar barra original con dos
    g.remove_edge(n1, n2)
    g.add_edge(n1, new_id, L1)
    g.add_edge(new_id, n2, L2)

    # Agregar barra extra usando distancias reales si tenemos posiciones
    candidates = [n['id'] for n in g.nodes
                  if n['id'] not in (n1, n2, new_id) and
                  g.node_degree(n['id']) < 5]
    if candidates and positions:
        target = rng.choice(candidates)
        # Estimar posición del nuevo pivote como interpolación
        if n1 in positions and n2 in positions:
            p1 = positions[n1]
            p2 = positions[n2]
            px = p1[0] + (p2[0] - p1[0]) * ratio
            py = p1[1] + (p2[1] - p1[1]) * ratio
            if target in positions:
                pt = positions[target]
                import math
                dist = math.sqrt((px - pt[0])**2 + (py - pt[1])**2)
                # Usar distancia real con perturbación pequeña
                extra_L = dist * rng.uniform(0.85, 1.15)
                g.add_edge(new_id, target, max(10.0, extra_L))
                return
        # Fallback: usar proporción de L
        extra_L = L * rng.uniform(0.4, 1.2)
        g.add_edge(new_id, target, max(10.0, extra_L))
    elif candidates:
        target = rng.choice(candidates)
        extra_L = L * rng.uniform(0.5, 1.5)
        g.add_edge(new_id, target, max(10.0, extra_L))


def _add_bar(g, rng):
    """Conecta dos pivotes no conectados, usando distancia real si posible."""
    node_ids = [n['id'] for n in g.nodes]
    existing = {(e['n1'], e['n2']) for e in g.edges}
    existing |= {(e['n2'], e['n1']) for e in g.edges}

    candidates = []
    for i in node_ids:
        for j in node_ids:
            if i < j and (i, j) not in existing:
                candidates.append((i, j))

    if not candidates:
        return

    # Resolver para obtener distancias reales
    from general_solver import solve_linkage
    positions = solve_linkage(g.nodes, g.edges, rng.uniform(0, 360))

    n1, n2 = rng.choice(candidates)

    if positions and n1 in positions and n2 in positions:
        import math
        p1, p2 = positions[n1], positions[n2]
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        new_L = dist * rng.uniform(0.8, 1.2)  # near-real distance
    else:
        avg_L = np.mean([e['length'] for e in g.edges]) if g.edges else 500.0
        new_L = avg_L * rng.uniform(0.3, 2.0)

    g.add_edge(n1, n2, max(10.0, new_L))


def _remove_bar(g, rng):
    """Elimina una barra si no desconecta el grafo."""
    if len(g.edges) <= MIN_EDGES:
        return

    shuffled = list(g.edges)
    rng.shuffle(shuffled)

    for e in shuffled:
        # No quitar la barra del crank
        crank_ids = [n['id'] for n in g.nodes if n['type'] == 'crank']
        ground_ids = [n['id'] for n in g.nodes if n['type'] == 'ground']
        if (e['n1'] in crank_ids and e['n2'] in ground_ids) or \
           (e['n2'] in crank_ids and e['n1'] in ground_ids):
            continue

        # Testear si la eliminación desconecta
        backup = g.edges[:]
        g.remove_edge(e['n1'], e['n2'])
        if g.is_connected():
            return  # Éxito
        else:
            g.edges = backup  # Restaurar


def _remove_pivot(g, rng):
    """Elimina un pivote libre con degree=2 (fusiona las dos barras)."""
    candidates = [n for n in g.nodes
                  if n['type'] == 'free' and g.node_degree(n['id']) == 2]
    if not candidates:
        return

    node = rng.choice(candidates)
    nid = node['id']
    nb = g.neighbors(nid)
    if len(nb) != 2:
        return

    (n1, l1), (n2, l2) = nb

    # Fusionar: eliminar nodo y barras, agregar barra directa
    g.remove_edge(nid, n1)
    g.remove_edge(nid, n2)
    g.nodes = [n for n in g.nodes if n['id'] != nid]

    new_L = l1 + l2  # o promedio, según estrategia
    g.add_edge(n1, n2, new_L * rng.uniform(0.7, 1.0))

    # Actualizar foot/knee si era ese nodo
    if g.foot_id == nid:
        g.foot_id = n2
    if g.knee_id == nid:
        g.knee_id = n1


def _reassign_output(g, rng):
    """Cambia qué nodo es foot o knee."""
    free = [n for n in g.nodes if n['type'] in ('free', 'output_foot', 'output_knee')]
    if len(free) < 2:
        return

    target = rng.choice(['foot', 'knee'])
    new_node = rng.choice(free)

    # Reset old
    for n in g.nodes:
        if target == 'foot' and n['type'] == 'output_foot':
            n['type'] = 'free'
        elif target == 'knee' and n['type'] == 'output_knee':
            n['type'] = 'free'

    if target == 'foot':
        new_node['type'] = 'output_foot'
        g.foot_id = new_node['id']
    else:
        new_node['type'] = 'output_knee'
        g.knee_id = new_node['id']


# ═══════════════════════════════════════════════════════════════════════════════
# CROSSOVER
# ═══════════════════════════════════════════════════════════════════════════════

def crossover(parent1, parent2, rng=None):
    """Crossover de dos genomas: toma la estructura de parent1 con longitudes
    interpoladas de parent2 donde sea posible."""
    if rng is None:
        rng = random.Random()

    child = parent1.copy()
    child.fitness = None

    # Para cada barra del child, si parent2 tiene una similar, interpolar
    p2_lengths = {}
    for e in parent2.edges:
        key = (min(e['n1'], e['n2']), max(e['n1'], e['n2']))
        p2_lengths[key] = e['length']

    alpha = rng.uniform(0.3, 0.7)
    for e in child.edges:
        key = (min(e['n1'], e['n2']), max(e['n1'], e['n2']))
        if key in p2_lengths:
            e['length'] = alpha * e['length'] + (1 - alpha) * p2_lengths[key]

    # A veces también interpolar ground positions
    if rng.random() < 0.3:
        for n1 in child.nodes:
            if n1['type'] == 'ground':
                for n2 in parent2.nodes:
                    if n2['type'] == 'ground' and n2['id'] == n1['id']:
                        n1['x'] = alpha * n1['x'] + (1 - alpha) * n2['x']
                        n1['y'] = alpha * n1['y'] + (1 - alpha) * n2['y']

    return child
