from Simulator import *
import numpy as np
import torch
from Env import Yahtzee
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


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
    def calcUCT(self, node, c = 0.08):
        UCT_values = {}
        N_total = node.N
        Qs = {}
        Us = {}

        for idx in node.child.keys():
            w = node.child[idx].W
            n = node.child[idx].N
            Qs[idx] = w/n
            Us[idx] = c * np.sqrt(np.log(N_total) / n)
            UCT_values[idx] = Qs[idx] + Us[idx]
        return UCT_values

    def selection(self, node):
        UCT_values = self.calcUCT(node)
        max_idx = max(UCT_values, key = UCT_values.get)
        return node.child[max_idx]
        
    def expansion(self, node):
        move = np.random.choice(node.untried_moves)
        state_, _, _, _ = step_with_int(node.state, move)
        node.untried_moves.remove(move)
        
        child_node = Node(node, move, state_)
        node.child[move] = child_node    
        return child_node

    def sampleState(self, node):
        act_history = []
        node.getPath(act_history)
        state = self.root_node.state        
        for act in act_history:
            state,_,_,_ = step_with_int(state, act)
        return state

    def evalState(self, node, net):
        done = not node.state["table"].count(None)            
        state = self.sampleState(node)
        if done:
            value = calcOutcome(state)
        else:
            if net is not None:
                state = preprocessing(state)
                state = torch.tensor(state, dtype = torch.float).cuda().unsqueeze(0)
                with torch.no_grad():
                    value = net(state)
                    value = value.squeeze().item()            
            else:
                state_ = state
                while not done:
                    legal_actions= getLegalAction(state_["left_rollout"], state_["table"])
                    action = np.random.choice(legal_actions)
                    state_,_,done,_ = step_with_int(state_, action)
                value = calcOutcome(state_)                            
        return value


    def backpropagation(self, node, value):
        node.W += value
        node.N += 1
        if not node.isRoot():
            self.backpropagation(node.parent, value)


    def search(self, n_sim, root_state, net = None):
        self.root_node = Node(None, None, root_state)
        
        for _ in range(n_sim):
            node = self.root_node
            
            while not node.isLeaf():
                node = self.selection(node)                

            if node.terminal:
                state = self.sampleState(node)
                leaf_value = calcOutcome(state)
                self.backpropagation(node, leaf_value)
            else:
                expanded_node = self.expansion(node)
                expanded_value = self.evalState(expanded_node, net)
                self.backpropagation(expanded_node, expanded_value)
            
        childs = self.root_node.child
        #visits = {k: v.N     for k, v in childs.items()}
        values = {k: v.W/v.N for k, v in childs.items()}
        action = max(values, key = values.get)
        return action


def main(n_sim, n_episode):
    score_list = []    
    mcts = MCTS()
    for e in range(n_episode):        
        done = False
        state = env.reset()        
        score = 0        
        while not done:
            action = mcts.search(n_sim, state)
            if render:
                env.render()
                if action < 31:
                    str_act = int2rollout[action]
                else:
                    str_act = int2input[action]
                print(str_act, "\n")

            state, reward, done, _ = env.step(action)    
            score += reward                            
        #done                
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Average: {average_score:.1f}")
    return average_score



render = False
env = Yahtzee()

if __name__ == "__main__":
    n_episode = 100
    avg_scores = []
    n_sims = range(100, 1001, 100)
    for n_sim in n_sims:
        avg_score = main(n_sim, n_episode)
        avg_scores.append(avg_score)    
    plt.scatter(n_sims, avg_scores)
    plt.savefig("save.png")
    plt.show()
        