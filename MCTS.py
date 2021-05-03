from Simulator import *
import numpy as np
import torch



class Node:
    def __init__(self, parent, move_idx, state_dict):
        self.parent = parent
        self.state = state_dict
        self.move_idx = move_idx
        self.W = 0
        self.N = 0
        self.child = {}
        self.legal_moves = getLegalAction(state_dict["left_rollout"],
                                        state_dict["table"])
        self.untried_moves = self.legal_moves.copy()
        self.terminal = len(self.legal_moves) == 0

    def isLeaf(self):
        return self.untried_moves != []
        
    def isRoot(self):
        return self.parent is None
        
    def getPath(self, path_list):
        if not self.isRoot():
            path_list.insert(0, self.move_idx)
            self.parent.getPath(path_list)
        



class MCTS:
    def __init__(self, net):
        self.net = net

    def reset(self, state):
        self.root_node = Node(None, None, state)
        self.root_state = state
        
    def calcUCT(self, node, c_puct = 0.005):
        UCT_values = {}
        Qs = {}
        Us = {}
        N_total = sum(node.N.values())

        for idx in node.child.keys():
            w = node.W[idx]
            n = node.N[idx]
            Qs[idx] = w/n
            Us[idx] = c_puct  * np.sqrt(np.log(N_total) / n)
            UCT_values[idx] = Qs[idx] + Us[idx]
        return UCT_values

    def selection(self, node):
        UCT_values = self.calcUCT(node)
        max_idx = max(UCT_values, key = UCT_values.get)
        return node.child[max_idx]
        
    def expansion(self, node, state):
        move = np.random.choice(node.untried_moves)
        state_, _, _, _ = step_with_int(state, move)
        node.untried_moves.remove(move)
        
        child_node = Node(node, move, state_)
        node.child[move] = child_node    
        return child_node, state_

    def rollout(self, child_node, state, k = 10):
        '''
        1. Move to leaf state follow action history
        2. Start simulation from leaf state
        3. Calc average score via value network
        '''
        values = []
        state_batch = []
        for _ in range(k):
            #sim to child grid                                            
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
            if node.move_idx in node.parent.W.keys():
                node.parent.W[node.move_idx] += value
                node.parent.N[node.move_idx] += 1
            else:
                node.parent.W[node.move_idx] = value
                node.parent.N[node.move_idx] = 1
            self.backpropagation(node.parent, value)


    def search(self, n_sim, root_state):
        self.root_node = Node(None, None, root_state)
        
        for _ in range(n_sim):
            node = self.root_node
            state = root_state

            while not node.isLeaf():
                node = self.selection(node)
                state = step_with_int(state, node.move_idx)

            if node.terminal:
                leaf_value = calcOutcome(state)
                self.backpropagation(node, leaf_value)                
            else:
                expanded_node, state = self.expansion(node, state)
                expanded_value = self.rollout(expanded_node, state)
                self.backpropagation(expanded_node, expanded_value)
    
        
        action = max(self.root_node.child.values, key = lambda x : x.N).move_idx
        return action
    