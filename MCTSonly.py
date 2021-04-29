from Simulator import *
import numpy as np
from Env import Yahtzee



class MCTS:
    def __init__(self):
        pass
    
    def setRoot(self, state):
        self.root_state = state
        self.values = [0] * 32
        self.visits = [1e-8] * 32

    def simulation(self, n_sim):                
        #주사위남음
        if self.root_state["left_rollout"] != 0:            
            for _ in range(n_sim):
                first_action = np.random.randint(32)
                if first_action != 31:
                    state,_,_,_ = step_with_int(self.root_state, first_action)
                    gap = self.simulDice(state)
                    value = max(gap)
                else:                    
                    gap = self.eval(self.root_state)
                    value = max(gap)
                self.values[first_action] += value
                self.visits[first_action] += 1
            mean_vals = [a/b for a,b in zip(self.values, self.visits)]
            action = np.argmax(mean_vals)            
            if action != 31:
                return action
                
        legal_moves = getLegalAction(0, self.root_state["table"])
        gap = self.eval(self.root_state)
        act_val = {k:gap[k-31] for k in legal_moves}        
        return max(act_val, key= act_val.get)
        
    def eval(self, state):
        reward_table = calcScoretable(state["dice"], state["table"])            
        gap = [b - a for a,b in zip(expectation, reward_table)]   
        return gap

    def simulDice(self, state):
        # 재귀적으로 주사위 시뮬레이션        
        if state["left_rollout"] == 0:            
            return self.eval(state)
        action = np.random.randint(31)
        new_state,_,_,_ = step_with_int(state, action)
        return self.simulDice(new_state)
        

    def doAction(self, state):
        legal_actions = getLegalAction(state["left_rollout"],
                                            state["table"])
        action = np.random.choice(legal_actions)
        return step_with_int(state, action)




def convertAction(int_action):
    if int_action < 31:
        act = int2rollout[int_action]
    else:
        act = int2input[int_action] + 31
    return act


def main():
    n_episode = 1
    n_sim = 10000
    env = Yahtzee()
    mcts = MCTS()
    score_list = []    
    for e in range(n_episode):        
        done = False
        state = env.reset()
        score = 0
        while not done:
            mcts.setRoot(state)
            env.render()
            action = mcts.simulation(n_sim)                        
            print(convertAction(action))
            state, reward, done, _ = env.step(action)    
            score += reward
        #done        
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Average: {average_score:.1f}")

if __name__ == "__main__":
    main()