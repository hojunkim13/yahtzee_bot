from Simulator import *
import numpy as np
from Env import Yahtzee



class MCTS:
    def __init__(self):
        pass
    
    def setRoot(self, state):
        self.root_state = state
        self.values = [0] * 44
        self.visits = [1e-8] * 44

    def simulation(self, n_sim, k):                
        #주사위남음
        legal_moves = getLegalAction(self.root_state["left_rollout"],
                                    self.root_state["table"])        
        for _ in range(n_sim):
            first_action = np.random.choice(legal_moves)
            state,_, done,_ = step_with_int(self.root_state, first_action)
            if done:
                value = calcOutcome(state)
            else:
                value = self.rollout(state, k)
            self.values[first_action] += value
            self.visits[first_action] += 1
        mean_values = [a / b for a,b in zip(self.values, self.visits)]
        best_act = np.argmax(mean_values)
        return best_act
        
    def rollout(self, target_state, k):
        done = False
        values = []
        for _ in range(k):
            state = target_state
            while not done:
                state, _, done, _ = self.doAction(state)
            value = calcOutcome(state)
            values.append(value)
            done = False
        return np.mean(values)

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
    n_sim = 100
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
            action = mcts.simulation(n_sim, 1000)
            print(convertAction(action),"\n")
            state, reward, done, _ = env.step(action)    
            score += reward
        #done        
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Average: {average_score:.1f}")

if __name__ == "__main__":
    main()