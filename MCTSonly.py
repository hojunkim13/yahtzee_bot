from Simulator import *
import numpy as np


from Env import Yahtzee


n_sim = 1000
k = 50
n_episode = 10
render = False
env = Yahtzee()


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

    def rollout(self, state, k = 10):
        '''
        1. Move to leaf state follow action history
        2. Start simulation from leaf state
        3. Calc average score via value network
        '''
        values = []        
        for _ in range(k):
            done = not state["table"].count(None)
            state_ = state
            #sim to child grid
            while not done:
                legal_moves = getLegalAction(state_["left_rollout"],
                                                state_["table"])
                move = np.random.choice(legal_moves)                                            
                state_,_,done,_ = step_with_int(state_, move)
            value = calcOutcome(state_)
            values.append(value)
        return np.mean(values)


    def backpropagation(self, node, value):
        node.W += value
        node.N += 1
        if not node.isRoot():
            self.backpropagation(node.parent, value)


    def search(self, n_sim, k, root_state):
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
                expanded_value = self.rollout(state, k)
                self.backpropagation(expanded_node, expanded_value)
            
        childs = self.root_node.child
        visits = {k: v.N     for k, v in childs.items()}
        values = {k: v.W/v.N for k, v in childs.items()}
        action = max(values, key = values.get)
        return action
    


def main():
    score_list = []    
    mcts = MCTS()
    for e in range(n_episode):        
        done = False
        state = env.reset()        
        score = 0        
        while not done:
            action = mcts.search(n_sim, k, state)
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

if __name__ == "__main__":
    main()