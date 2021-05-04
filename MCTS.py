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
        return self.terminal or self.untried_moves != []
        
    def isRoot(self):
        return self.parent is None
        
    def getPath(self, path_list):
        if not self.isRoot():
            path_list.insert(0, self.move_idx)
            self.parent.getPath(path_list)
        

class MCTS:
    def calcUCT(self, node, c_puct = 0.01):
        UCT_values = {}
        N_total = node.N
        Qs = {}
        Us = {}

        for idx in node.child.keys():
            w = node.child[idx].W
            n = node.child[idx].N
            Qs[idx] = w/n
            Us[idx] = c_puct * np.sqrt(np.log(N_total) / n)
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

    def evalState(self, state, net):
        done = not state["table"].count(None)            
        if done:
            value = calcOutcome(state)
        else:
            state = preprocessing(state)
            state = torch.tensor(state, dtype = torch.float).cuda().unsqueeze(0)
            with torch.no_grad():
                value = net(state)
                value = value.squeeze().item()            
        return value


    def backpropagation(self, node, value):
        node.W += value
        node.N += 1
        if not node.isRoot():
            self.backpropagation(node.parent, value)


    def search(self, n_sim, net, root_state):
        self.root_node = Node(None, None, root_state)
        
        for _ in range(n_sim):
            node = self.root_node
            state = root_state

            while not node.isLeaf():
                node = self.selection(node)
                state,_,_,_ = step_with_int(state, node.move_idx)

            if node.terminal:
                leaf_value = calcOutcome(state)
                self.backpropagation(node, leaf_value)                
            else:
                expanded_node, state = self.expansion(node, state)
                expanded_value = self.evalState(state, net)
                self.backpropagation(expanded_node, expanded_value)
            
        childs = self.root_node.child
        #visits = {k: v.N     for k, v in childs.items()}
        values = {k: v.W/v.N for k, v in childs.items()}
        action = max(values, key = values.get)
        return action
    