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
        self.child = {}
        self.legal_moves = getLegalAction(state_dict["left_rollout"],
                                        state_dict["table"])
        if len(self.legal_moves) == 0:
            self.terminal = True
        else:
            self.terminal = False
        
            
    def calcUCT(self, c_puct = 0.005):
        UCT_values = {}
        Qs = {}
        Us = {}
        N_total = sum(self.N.values())

        for idx in self.child.keys():
            w = self.W[idx]
            n = self.N[idx]
            Qs[idx] = w/n
            Us[idx] = c_puct  * np.sqrt(np.log(N_total) / n)
            UCT_values[idx] = Qs[idx] + Us[idx]
        return UCT_values
    
    def calcVal(self):
        Qs = {}
        for idx in self.child.keys():
            w = self.W[idx]
            n = self.N[idx]
            Qs[idx] = w/n
        
        return Qs

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
        
    def getPath(self, path_list):
        if not self.isRoot():
            path_list.insert(0, self.move_index)
            self.parent.getPath(path_list)
        


class MCTS:
    def __init__(self, net):
        self.net = net

    def reset(self, state):
        self.root_node = Node(None, None, state)
        self.root_state = state
        
    def selection(self, node):
        if node.isLeaf():
            return node
        else:
            UCT_values = node.calcUCT()            
            child_idx = max(UCT_values, key=UCT_values.get)            
            node = node.child[child_idx]
            return self.selection(node)
        

    def expansion(self, node):
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
        if len(state_batch) != 0:
            state_batch = torch.tensor(state_batch, dtype = torch.float).cuda()
            with torch.no_grad():
                values_ = self.net(state_batch)
            values_ = torch.clip(values_, 0, None).sum().cpu().item()
        else:
            values_ = 0
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
            leaf_value = calcOutcome(leaf_node.state)
            self.backpropagation(leaf_node, leaf_value)

    def search(self, n_sim):
        for _ in range(n_sim):
            self.search_cycle()

        val = self.root_node.calcVal()
        action = max(val, key = val.get)
        return action
    