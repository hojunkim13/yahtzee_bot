from Simulator import *
import numpy as np
import torch



class Node:
    def __init__(self, parent, move_index, state_dict):
        self.parent = parent
        self.state = state_dict
        self.move_index = move_index
        self.W = {}
        self.N = {}
        self.P = {}
        self.child = {}
        self.legal_moves = getLegalAction(state_dict["left_rollout"],
                                        state_dict["table"])
        if len(self.legal_moves) == 0:
            self.terminal = True
        else:
            self.terminal = False
        self.calcProb()
        
    def calcProb(self, net = None):
        if net is None:
            for moves in self.legal_moves:
                self.P[moves] = 1 / len(self.legal_moves)
        elif self.isRoot():
            state = preprocessing(self.state)
            state = torch.tensor(state, dtype = torch.float).cuda().unsqueeze(0)
            with torch.no_grad():
                probs, _ = net(state)
                probs = probs.squeeze().cpu().numpy()
        
            for moves in self.legal_moves:
                self.P[moves] = probs[moves]            
        else:
            raise ValueError
            

    def calcPUCT(self, c_puct = 1):
        PUCT_values = {}
        Qs = {}
        Us = {}
        N_total = sum(self.N.values())

        for idx in self.child.keys():
            w = self.W[idx]
            n = self.N[idx]
            p = self.P[idx]
            Qs[idx] = w/n
            Us[idx] = c_puct * (p/n) * np.sqrt(N_total) / (1 + n)
            PUCT_values[idx] = Qs[idx] + Us[idx]
        return PUCT_values

    def isLeaf(self):
        if self.terminal:
            return True
        for move in self.legal_moves:        
            try:
                self.child[move]
            except KeyError:
                return True        
        return False
        

    def isRoot(self):
        return self.parent is None

    def asRoot(self):
        self.parent = None
        self.move_index = None
        
    def getPath(self, path_list):
        if not self.isRoot():
            path_list.insert(0, self.move_index)
            self.parent.getPath(path_list)
        


class MCTS:
    def __init__(self, net):
        self.net = net

    def reset(self, state):
        self.root_node = Node(None, None, state)
        self.root_node.calcProb(self.net)
        self.root_state = state
        
    def selection(self, node):
        if node.isLeaf():
            return node
        else:
            PUCT_values = node.calcPUCT()            
            max_value_idx = max(PUCT_values, key=PUCT_values.get)
            
            node = node.child[max_value_idx]
            return self.selection(node)
        

    def expansion(self, node):
        if node.terminal:
            return 
        left_indice = list(set(node.legal_moves) - (set(node.child.keys())))
        idx = np.random.choice(left_indice)
        state_, _, _, _ = step_with_int(node.state, idx)
        child_node = Node(node, idx, state_)
        node.child[idx] = child_node
        return child_node

    def simulation(self, child_node, k = 10):
        '''
        1. Move to leaf state follow action history
        2. Start simulation from leaf state
        3. Calc average score via value network
        '''
        values = []
        state_batch = []
        act_history = []
        #vs = []
        child_node.getPath(act_history)
        
        for _ in range(k):
            #sim to child grid
            state = self.root_state
            for act in act_history:
                state, _, done, _ = step_with_int(state, act)
            if done:
                value = calcOutcome(state)
                values.append(value)
            else:
                state_ppd = preprocessing(state)                
                state_batch.append(state_ppd)                            
        state_batch = torch.tensor(state_batch, dtype = torch.float).cuda().view(-1,22,6)
        with torch.no_grad():
            _, values_ = self.net(state_batch)
        values_ = torch.clip(values_, 0, None).sum().cpu().item()
        mean_value = (sum(values) + values_ ) / k
        return mean_value


    def backpropagation(self, node, value):                
        if not node.isRoot():
            if node.move_index in node.parent.W.keys():
                node.parent.W[node.move_index] += value
                node.parent.N[node.move_index] += 1
            else:
                node.parent.W[node.move_index] = value
                node.parent.N[node.move_index] = 1
            self.backpropagation(node.parent, value)
                
    def search_cycle(self):
        leaf_node = self.selection(self.root_node)
        if not leaf_node.terminal:
            expanded_node = self.expansion(leaf_node)
            expanded_value = self.simulation(expanded_node)
            self.backpropagation(expanded_node, expanded_value)
        else:
            value = self.simulation(leaf_node)
            self.backpropagation(leaf_node, value)

    def search(self, n_sim):
        for _ in range(n_sim):
            self.search_cycle()        
            
        return deepcopy(self.root_node.N)
    


# def main():
#     n_episode = 100
#     n_sim = 400
#     env.goal = 999999
#     mcts = MCTS()
#     score_list = []
#     for e in range(n_episode):
#         start_time = time.time()
#         done = False
#         score = 0
#         grid = env.reset()
#         mcts.setRoot(grid)        
#         while not done:        
#             action = mcts.getAction(n_sim)
#             grid, reward, done, info = env.step(action)
#             score += reward
#             mcts.setRoot(grid, action)            
#         mcts.saveMemory(e)
#         score_list.append(score)
#         average_score = np.mean(score_list[-100:])        
#         spending_time = time.time() - start_time
#         print(f"Episode : {e+1} / {n_episode}, Score : {score}, Max Tile : {info}, Average: {average_score:.1f}")
#         print(f"SPENDING TIME : {spending_time:.1f} Sec")
#     env.close()

# if __name__ == "__main__":
#     main()
