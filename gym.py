import sys
from numpy import *
import random as pyrandom
from collections import defaultdict

rng = random.default_rng()

sys.path.pop(0)

import ale_py
import gymnasium as gym

global inv,cnt

inv = 0
cnt = 0

#treba dodat biaseve


gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="ram", render_mode = "None",repeat_action_probability=0.0)
env = gym.wrappers.FlattenObservation(env)


class Model:
    def __init__(self, ulaz):
        global inv, cnt
        self.ulaz = ulaz
       
        # biases = []
        #6 clanova: od kud, do kud, weight, radi li, poredak, tip nodea
        weights = [] #connection matrica, ali da ne treba mijenjat sve 

        #4 clanova: koji ulazi, weight, active, innovation
        graf = defaultdict(list)
        #prev = ulaz
        cnt = len(ulaz)
        for i in range(len(ulaz)):
            for j in range(9):
                x = rng.normal(0, 1)
                weights.append([i,cnt+j,x,1,inv, "out"])
                graf[cnt+j].append([i,x,1,inv])
                inv+=1
            #cnt += 1
        cnt += 9
        
        code = inv
        self.weights = weights
        self.graf = graf
        self.score = 0
        self.cnt = cnt
        self.inv = inv
        self.topo_order = self.topo_sort() 
        #self.biases = biases

    def topo_sort(self):
        # in-degree = number of active incoming edges per node
        in_degree = defaultdict(int)
        successors = defaultdict(list)  # from_node -> [to_node, to_node, ...]

        all_nodes = set()
        for w in self.weights:
            frm, to, weight, active, innov, typ = w
            all_nodes.add(frm)
            all_nodes.add(to)
            if active == 1:
                in_degree[to] += 1
                successors[frm].append(to)

        # nodes with in_degree 0 = inputs + anything with no active incoming edge
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
       # ulaz = ulaz / 127.5 - 1.0
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

        return ans[n_inputs:n_inputs+9]

    def cross(self, x1, x2, sigma):
        global inv, cnt
        self.sigma = sigma

        weightx = []
        graphx = defaultdict(list)
        #biasex = []
        ci = 0
        cj = 0
        #cnt = 0

        for i in range (max(len(x1.weights),len(x2.weights))):
            w1 = []
            w2 = []
            x = 0
            if (ci <= len(x1.weights)-1):
                w1 = x1.weights[ci]
                ci += 1
            if (cj <= len(x2.weights)-1):
                w2 = x2.weights[cj] 
                cj += 1

            if not w1 and not w2:
                break
            elif not w1:
                nw = list(w2)
            elif not w2:
                nw = list(w1)
            else:
                if w1[4] != w2[4]:
                    if x1.score > x2.score:
                        weightx.append(w1)
                        graphx[w1[1]].append([w1[0], w1[1], w1[2], w1[3]])
                        continue   
                    elif x2.score > x1.score:
                        weightx.append(w2)
                        graphx[w2[1]].append([w2[0], w2[2], w2[3], w2[4]])
                        continue
                
                nw = list(pyrandom.choice([w1,w2]))
                x = rng.integers(0,100)
            if x < 10:
                nw[2] = nw[2] + rng.normal(0,sigma)
            elif x < 15:
                br1 = pyrandom.choice(weightx)
                br2 = pyrandom.choice(weightx)
                if (br1[0] == br2[0] or  br1[len(br1)-1] == "out"):
                    weightx.append(nw)
                    graphx[nw[1]].append([nw[0],nw[2],nw[3],nw[4]])
                    continue
                            
                kk = rng.normal(0,1)
                weightx.append([br1[0], br2[0], kk, 1, inv, br2[len(br2)-1]])
                graphx[br2[0]].append([br1[0], kk, 1, inv])
                inv+=1
            elif x < 18:
                #br1 = random.choice(nw)
                weightx.append([nw[0],cnt, 1.0 ,1, inv, "hidden"])
                weightx.append([cnt,nw[1], nw[2],1, inv+1, "hidden"])
                graphx[cnt].append([nw[0], 1.0, 1, inv])
                inv += 1
                graphx[nw[1]].append([cnt, nw[2], 1, inv])
                nw[3] = 0
                cnt+=1
                inv += 1    
                weightx.append(nw)
                continue

            weightx.append(nw)
            graphx[nw[1]].append([nw[0],nw[2],nw[3],nw[4]])

        # for b1, b2 in zip(x1.biases, x2.biases):
        #    nb = (b1 + b2) / 2
        #    mask = rng.random(nb.shape) > 0.94
        #   nb += mask * rng.normal(0, sigma, nb.shape)
        #   biasex.append(nb)

        self.weights = weightx
        self.graf = graphx
        self.cnt = cnt
        self.inv = inv
        self.topo_order = self.topo_sort()
        # self.biases = biasex


    
    
    

 
    def eval_model(self):
        obs, info = env.reset()
        poc = info["lives"]
        result = 0
        prev_obs = obs.copy()
        while True:
            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = env.step(action)

            if reward == 0:
                reward-=1
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
