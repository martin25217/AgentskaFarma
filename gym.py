import sys
from numpy import *
import random as pyrandom

rng = random.default_rng()

sys.path.pop(0)

import ale_py
import gymnasium as gym

# global, run-wide counters shared across the whole population
inv = 0
cnt = 0

# treba dodat biaseve

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="rgb_array", repeat_action_probability=0.0)
env = gym.wrappers.FlattenObservation(env)


class Model:
    def __init__(self, ulaz):
        global inv, cnt
        self.ulaz = ulaz

        # 5 clanova: od kud, do kud, weight, radi li, poredak (innovation), tip nodea
        weights = []

        num_inputs = len(ulaz)
        self.input_ids = list(range(num_inputs))
        self.output_ids = list(range(num_inputs, num_inputs + 9))
        self.hidden_ids = []

        cnt = num_inputs + 9  # next free node id
        graf = [[] for _ in range(cnt)]

        for i in self.input_ids:
            for j in self.output_ids:
                x = rng.normal(0, 1)
                weights.append([i, j, x, 1, inv, "out"])
                graf[j].append([i, x, 1, inv])
                inv += 1

        self.weights = weights
        self.graf = graf
        self.score = 0
        self.cnt = cnt
        # self.biases = biases

    def all_node_ids(self):
        return self.input_ids + self.output_ids + self.hidden_ids

    def rebuild_graf(self):
        # size graf dynamically so it always has room for the highest node id seen so far
        all_ids = self.all_node_ids()
        max_id = max(all_ids) if all_ids else 0
        graf = [[] for _ in range(max_id + 1)]
        for gene in self.weights:
            from_node, to_node, weight, enabled, innov = gene[0], gene[1], gene[2], gene[3], gene[4]
            graf[to_node].append([from_node, weight, enabled, innov])
        self.graf = graf

    def cross(self, x1, x2, sigma):
        global inv, cnt
        self.sigma = sigma
        self.score = 0

        # child inherits its parents' node registries (union, since either parent
        # may have grown hidden nodes the other doesn't have)
        self.input_ids = list(x1.input_ids)
        self.output_ids = list(x1.output_ids)
        self.hidden_ids = sorted(set(x1.hidden_ids) | set(x2.hidden_ids))

        weightx = []
        ci = 0
        cj = 0

        for i in range(max(len(x1.weights), len(x2.weights))):
            w1 = None
            w2 = None
            if ci <= len(x1.weights) - 1:
                w1 = x1.weights[ci]
                ci += 1
            if cj <= len(x2.weights) - 1:
                w2 = x2.weights[cj]
                cj += 1

            if w1 is None:
                nw = list(w2)
            elif w2 is None:
                nw = list(w1)
            else:
                if w1[4] != w2[4]:
                    # mismatched innovation numbers -> inherit from the fitter PARENT genome
                    if x1.score > x2.score:
                        nw = list(w1)
                    elif x2.score > x1.score:
                        nw = list(w2)
                    else:
                        nw = list(pyrandom.choice([w1, w2]))
                else:
                    nw = list(pyrandom.choice([w1, w2]))

            x = rng.integers(0, 100)
            if x < 10:
                # weight mutation
                nw[2] = nw[2] + rng.normal(0, sigma)

            elif x < 15:
                # new connection
                br1 = pyrandom.choice(self.all_node_ids())
                br2 = pyrandom.choice(self.all_node_ids())
                if br1 == br2 or br2 in self.input_ids:
                    weightx.append(nw)
                    continue
                weightx.append([br1, br2, rng.normal(0, 1), 1, inv, "hidden"])
                inv += 1

            elif x < 18:
                # new node (split an existing connection)
                new_node_id = cnt
                cnt += 1
                self.hidden_ids.append(new_node_id)

                weightx.append([nw[0], new_node_id, 1.0, 1, inv, "hidden"])
                inv += 1
                weightx.append([new_node_id, nw[1], nw[2], 1, inv, "hidden"])
                inv += 1
                nw[3] = 0  # disable the old connection

            weightx.append(nw)

        self.weights = weightx
        self.rebuild_graf()

        # for b1, b2 in zip(x1.biases, x2.biases):
        #     nb = (b1 + b2) / 2
        #     mask = rng.random(nb.shape) > 0.94
        #     nb += mask * rng.normal(0, sigma, nb.shape)
        #     biasex.append(nb)
        # self.biases = biasex

    def feed_forward(self, ulaz):
        ulaz = ulaz / 127.5 - 1.0
        graf = self.graf
        ans = zeros(9)

        for idx, node_id in enumerate(self.output_ids):
            node = graf[node_id] if node_id < len(graf) else []
            zb = 0
            for j in range(len(node)):
                if node[j][2] == 1:
                    zb += node[j][1] * ulaz[node[j][0]]
            ans[idx] = tanh(zb)

        return ans

    def eval_model(self):
        obs, info = env.reset()
        poc = info["lives"]
        result = 0
        prev_obs = obs.copy()
        while True:
            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = env.step(action)

            if reward == 0:
                reward -= 1
            if action == 0:
                reward = -1000
            if action != 0 and array_equal(obs, prev_obs):
                reward = -1000
            if reward > 100:
                reward = 50

            prev_obs = obs.copy()
            result = result + reward
            if terminated or truncated or info["lives"] < poc:
                break
        return result