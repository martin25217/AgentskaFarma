import sys
from numpy import *
import random as pyrandom
from collections import defaultdict, deque

rng = random.default_rng()

sys.path.pop(0)

import ale_py
import gymnasium as gym

global inv, cnt

inv = 0
cnt = 0

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="rgb", render_mode="rgb_arrayx   ", repeat_action_probability=0.0)

PACMAN_COLOR = array([210, 164, 74])
GHOST_COLORS = {
    "red":    array([200, 72, 72]),
    "pink":   array([198, 89, 179]),
    "cyan":   array([84, 184, 153]),
    "orange": array([180, 122, 48]),
}
GHOST_SCARED_COLOR = array([66, 114, 194]) 
PELLET_COLOR = array([187, 187, 53])
POWER_PELLET_COLOR = array([187, 187, 53])     

COLOR_TOLERANCE = 25

FRAME_H, FRAME_W = 210, 160
GRID_H, GRID_W = 8, 10

FRAME_STACK_SIZE = 4

# Feature layout (per frame):
#   pacman:        2   (x, y normalized)
#   4 ghosts:       4*3 = 12   (x, y, scared_flag)
#   power pellet:   3   (dx, dy to nearest, count_remaining/4)
#   pellet grid:    GRID_H*GRID_W
PER_FRAME_DIM = 2 + 12 + 3 + (GRID_H * GRID_W)
# Velocity features: pacman + 4 ghosts, (dx, dy) each, computed across the frame stack
VELOCITY_DIM = 2 + 4 * 2
INPUT_DIM = PER_FRAME_DIM + VELOCITY_DIM


def _color_mask(frame, color, tol=COLOR_TOLERANCE):
    return all(abs(frame.astype(int) - color) < tol, axis=-1)


def _centroid(mask):
    ys, xs = where(mask)
    if len(xs) == 0:
        return None
    return array([xs.mean(), ys.mean()])


def extract_frame_features(frame):
    """Returns a PER_FRAME_DIM vector, plus the raw (unnormalized) tracked
    positions needed for velocity calc: [pacman_xy, ghost1_xy, ..., ghost4_xy]."""

    feats = []
    raw_positions = []  # for velocity: pacman + 4 ghosts, in order

    # --- Pac-Man ---
    pac_c = _centroid(_color_mask(frame, PACMAN_COLOR))
    if pac_c is None:
        pac_c = array([FRAME_W / 2, FRAME_H / 2])  # fallback: center screen
    feats.append(pac_c[0] / FRAME_W)
    feats.append(pac_c[1] / FRAME_H)
    raw_positions.append(pac_c)

    # --- Ghosts ---
    for name, color in GHOST_COLORS.items():
        normal_mask = _color_mask(frame, color)
        scared_mask = _color_mask(frame, GHOST_SCARED_COLOR)
        c = _centroid(normal_mask)
        scared_flag = 0.0
        if c is None:
            c = _centroid(scared_mask)
            if c is not None:
                scared_flag = 1.0
        if c is None:
            c = array([FRAME_W / 2, FRAME_H / 2])  # not visible this frame
        feats.append(c[0] / FRAME_W)
        feats.append(c[1] / FRAME_H)
        feats.append(scared_flag)
        raw_positions.append(c)

    # --- Power pellets: nearest one relative to Pac-Man + how many remain ---
    pp_mask = _color_mask(frame, POWER_PELLET_COLOR)
    ys, xs = where(pp_mask)
    if len(xs) == 0:
        feats.extend([0.0, 0.0, 0.0])
    else:
        pts = stack([xs, ys], axis=1).astype(float)
        dists = ((pts[:, 0] - pac_c[0]) ** 2 + (pts[:, 1] - pac_c[1]) ** 2)
        nearest = pts[argmin(dists)]
        dx = (nearest[0] - pac_c[0]) / FRAME_W
        dy = (nearest[1] - pac_c[1]) / FRAME_H
        # crude "how many pellets remain" proxy via distinct-blob count would need
        # connected-components; using pixel count / expected-pixels-per-pellet as
        # a rough stand-in — tune the divisor once you know real pellet pixel size.
        approx_count = clip(len(xs) / 20.0, 0, 4) / 4.0
        feats.extend([dx, dy, approx_count])

    # --- Pellet density grid ---
    dot_mask = _color_mask(frame, PELLET_COLOR)
    cell_h = FRAME_H // GRID_H
    cell_w = FRAME_W // GRID_W
    for i in range(GRID_H):
        for j in range(GRID_W):
            cell = dot_mask[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            feats.append(cell.mean())

    return array(feats), raw_positions


def build_input_vector(frame_history):
    """frame_history: deque of (feats, raw_positions) tuples, most recent last.
    Returns the full INPUT_DIM vector: latest frame's features + velocity."""

    latest_feats, latest_positions = frame_history[-1]

    if len(frame_history) >= 2:
        oldest_positions = frame_history[0][1]
        n_steps = len(frame_history) - 1
        velocity = []
        for cur, old in zip(latest_positions, oldest_positions):
            velocity.append((cur[0] - old[0]) / FRAME_W / n_steps)
            velocity.append((cur[1] - old[1]) / FRAME_H / n_steps)
    else:
        velocity = [0.0] * VELOCITY_DIM

    return concatenate([latest_feats, array(velocity)])



    ####SVE GORE JE ZA GENERIRANJE INPUT VEKTORA

def compatibility_distance(g1, g2):

    genes1 = sorted(g1.weights, key=lambda x: x[4])
    genes2 = sorted(g2.weights, key=lambda x: x[4])

    i = 0
    j = 0

    matching = 0
    disjoint = 0
    excess = 0
    weight_diff = 0

    while i < len(genes1) and j < len(genes2):

        if genes1[i][4] == genes2[j][4]:

            matching += 1
            weight_diff += abs(genes1[i][2] - genes2[j][2])

            i += 1
            j += 1

        elif genes1[i][4] < genes2[j][4]:

            disjoint += 1
            i += 1

        else:

            disjoint += 1
            j += 1

    excess += len(genes1) - i
    excess += len(genes2) - j

    N = max(len(genes1), len(genes2))
    if N < 20:
        N = 1

    if matching > 0:
        W = weight_diff / matching
    else:
        W = 0

    return excess / N + disjoint / N + 0.4 * W

def napravi_species(populacija, threshold=0.5):

    species = []

    for genome in populacija:

        stavljen = False

        for grupa in species:

            predstavnik = grupa[0]

            if compatibility_distance(genome, predstavnik) < threshold:
                grupa.append(genome)
                stavljen = True
                break

        if not stavljen:
            species.append([genome])

    return species

class Model:
    def __init__(self, ulaz):
        global inv, cnt
        self.ulaz = ulaz

        weights = []
        graf = defaultdict(list)
        cnt = len(ulaz)
        for i in range(len(ulaz)):
            for j in range(9):
                x = rng.normal(0, 1)
                weights.append([i, cnt + j, x, 1, inv, "out"])
                graf[cnt + j].append([i, x, 1, inv])
                inv += 1
        cnt += 9

        self.weights = weights
        self.graf = graf
        self.score = 0
        self.cnt = cnt
        self.inv = inv
        self.topo_order = self.topo_sort()

    def topo_sort(self):
        in_degree = defaultdict(int)
        successors = defaultdict(list)

        all_nodes = set()
        for w in self.weights:
            frm, to, weight, active, innov, typ = w
            all_nodes.add(frm)
            all_nodes.add(to)
            if active == 1:
                in_degree[to] += 1
                successors[frm].append(to)

        queue = [n for n in all_nodes if in_degree[n] == 0]
        order = []

        while queue:
            node = queue.pop()
            order.append(node)
            for nxt in successors[node]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(all_nodes):
            raise ValueError("cycle detected in genome — not a valid feedforward network")

        return order
    

    def feed_forward(self, ulaz):
        # ulaz is already a normalized feature vector — no /127.5 pixel scaling needed
        graf = self.graf

        ans = zeros(self.cnt)
        ans[:len(ulaz)] = ulaz

        n_inputs = len(ulaz)
        for i in self.topo_order:
            if i < n_inputs:
                continue
            node = graf[i]
            zb = 0
            for j in range(len(node)):
                if node[j][2] == 1:
                    zb += node[j][1] * ans[node[j][0]]
            ans[i] = tanh(zb)

        return ans[n_inputs:n_inputs + 9]

    def actual_node_count(self):
        nodes = set(range(len(self.ulaz) + 9))

        for connection in self.weights:
            nodes.add(connection[0])
            nodes.add(connection[1])

        return len(nodes)

    
    def path(self,graph,n1,n2, visited = None):
        if visited is None:
            visited = set()
        if n2 in visited:
            return False
        visited.add(n2)
        for i in range(len(graf[n2])):
            if (graf[n2][i][0] == n1):
                return True
            
            if self.path(graph,n1, graf[n2][i][0], visited):
                return True
        return False

    def rebuild_graph(self):
        graph = defaultdict(list)

        for gene in self.weights:
            from_node = gene[0]
            to_node = gene[1]
            weight = gene[2]
            enabled = gene[3]
            innovation = gene[4]

            graph[to_node].append([
                from_node,
                weight,
                enabled,
                innovation
            ])

        self.graf = graph
    def mutate_add_connection(self, max_attempts=30):
        global inv

        input_count = len(self.ulaz)
        output_start = input_count
        output_end = input_count + 9

        nodes = set(range(output_end))

        for gene in self.weights:
            nodes.add(gene[0])
            nodes.add(gene[1])

        nodes = list(nodes)

        existing_connections = {
            (gene[0], gene[1])
            for gene in self.weights
        }

        for _ in range(max_attempts):
            from_node = pyrandom.choice(nodes)
            to_node = pyrandom.choice(nodes)

            if from_node == to_node or to_node < input_count or output_start <= from_node < output_end:
                continue

            if (from_node, to_node) in existing_connections:
                continue

            weight = rng.normal(0, 1)

            temporary_gene = [
                from_node,
                to_node,
                weight,
                1,
                inv,
                (
                    "out"
                    if output_start <= to_node < output_end
                    else "hidden"
                )
            ]

            # Privremeno dodaj vezu
            self.weights.append(temporary_gene)
            self.rebuild_graph()

            try:
                new_order = self.topo_sort()

            except ValueError:
                # Veza stvara ciklus, ukloni je
                self.weights.pop()
                self.rebuild_graph()
                continue

            # Veza je valjana
            self.topo_order = new_order
            inv += 1
            return True

        return False
        
    

    def cross(self, x1, x2, sigma):
        global inv, cnt

        self.ulaz = x1.ulaz.copy()
        self.score = 0
        self.sigma = sigma

        if x2.score > x1.score:
            fitter = x2
            weaker = x1
        else:
            fitter = x1
            weaker = x2

        equal_fitness = x1.score == x2.score

        genes1 = sorted(x1.weights, key=lambda gene: gene[4])
        genes2 = sorted(x2.weights, key=lambda gene: gene[4])

        child_genes = []

        i = 0
        j = 0

        while i < len(genes1) and j < len(genes2):

            gene1 = genes1[i]
            gene2 = genes2[j]

            innovation1 = gene1[4]
            innovation2 = gene2[4]

            if innovation1 == innovation2:
                new_gene = pyrandom.choice([gene1, gene2]).copy()

                # Ako je gen kod jednog roditelja disabled,
                # postoji velika šansa da ostane disabled
                if gene1[3] == 0 or gene2[3] == 0:
                    new_gene[3] = 0 if rng.random() < 0.75 else 1

                child_genes.append(new_gene)

                i += 1
                j += 1

            elif innovation1 < innovation2:

                if equal_fitness:
                    if rng.random() < 0.5:
                        child_genes.append(gene1.copy())

                elif fitter is x1:
                    child_genes.append(gene1.copy())

                i += 1

            else:

                if equal_fitness:
                    if rng.random() < 0.5:
                        child_genes.append(gene2.copy())

                elif fitter is x2:
                    child_genes.append(gene2.copy())

                j += 1

        while i < len(genes1):
            gene1 = genes1[i]

            if equal_fitness:
                if rng.random() < 0.5:
                    child_genes.append(gene1.copy())

            elif fitter is x1:
                child_genes.append(gene1.copy())

            i += 1

        while j < len(genes2):
            gene2 = genes2[j]

            if equal_fitness:
                if rng.random() < 0.5:
                    child_genes.append(gene2.copy())

            elif fitter is x2:
                child_genes.append(gene2.copy())

            j += 1

        self.weights = child_genes

        for gene in self.weights:
            if rng.random() < 0.10:
                gene[2] += rng.normal(0, sigma)

            # Mala šansa da se težina potpuno zamijeni
            elif rng.random() < 0.01:
                gene[2] = rng.normal(0, 1)

        if rng.random() < 0.03:
            enabled_genes = [
                gene for gene in self.weights
                if gene[3] == 1
            ]

            if enabled_genes:
                old_gene = pyrandom.choice(enabled_genes)

                old_from = old_gene[0]
                old_to = old_gene[1]
                old_weight = old_gene[2]

                # Isključi staru vezu
                old_gene[3] = 0

                new_node = cnt
                cnt += 1

                # old_from -> new_node
                self.weights.append([
                    old_from,
                    new_node,
                    1.0,
                    1,
                    inv,
                    "hidden"
                ])
                inv += 1

                # new_node -> old_to
                self.weights.append([
                    new_node,
                    old_to,
                    old_weight,
                    1,
                    inv,
                    old_gene[5]
                ])
                inv += 1

        # --------------------------------------------------
        # 5. Napravi privremeni graf
        # --------------------------------------------------

        self.rebuild_graph()

        # --------------------------------------------------
        # 6. Add-connection mutacija — jednom po genomu
        # --------------------------------------------------

        if rng.random() < 0.05:
            self.mutate_add_connection()

        # Ponovno napravi graf jer je možda dodana veza
        self.rebuild_graph()

        # --------------------------------------------------
        # 7. Završni podaci djeteta
        # --------------------------------------------------

        self.inv = inv
        self.cnt = max(
            [len(self.ulaz) + 9]
            + [
                max(gene[0], gene[1]) + 1
                for gene in self.weights
            ]
        )

        self.weights.sort(key=lambda gene: gene[4])
        self.topo_order = self.topo_sort()

    def eval_model(self):
        obs, info = env.reset()
        #feats, raw_pos = extract_frame_features(obs)
        #print("feats:", feats)
        #print("raw_pos:", raw_pos)
        poc = info["lives"]
        result = 0

        frame_history = deque(maxlen=FRAME_STACK_SIZE)
        feats, raw_pos = extract_frame_features(obs)
        frame_history.append((feats, raw_pos))
        prev_obs = obs.copy()

        while True:
            input_vec = build_input_vector(frame_history)
            action = int(argmax(self.feed_forward(input_vec)))
            obs, reward, terminated, truncated, info = env.step(action  )

            if reward == 0:
                reward -= 1
            if action == 0:
                reward = -5
            if action != 0 and array_equal(obs, prev_obs):
                reward = -5
            if reward > 100:
                reward = 50

            prev_obs = obs.copy()
            result = result + reward

            feats, raw_pos = extract_frame_features(obs)
            frame_history.append((feats, raw_pos))

            if terminated or truncated or info["lives"] < poc:
                break
        return result