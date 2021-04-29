import numpy as np
from Env import Yahtzee
from Agent import Agent
from collections import deque
from Simulator import *

lr = 1e-3
batch_size = 128
n_sim = 100
maxlen = 10000
n_episode = 10000
state_dim = (16,4,4)
action_dim = 4
agent = Agent(lr, batch_size, maxlen, n_sim)
env = Yahtzee()
#agent.load("Yahtzee")

def main():
    score_list = []    
    for e in range(n_episode):        
        done = False
        state = env.reset()
        tmp_memory = deque()
        score = 0
        agent.step_count = 0
        while not done:
            #env.render()
            agent.mcts.reset(state)
            action, probs = agent.getAction()
            tmp_memory.append((preprocessing(state), probs))
            if action not in agent.mcts.root_node.legal_moves:
                print("warning")
            state, reward, done, info = env.step(action)    
            score += reward            
        outcome = calcOutcome(state)
        agent.pushMemory(tmp_memory, outcome)
        loss = agent.learn()

        #done
        if (e+1) % 10 == 0:
            agent.save("Yahtzee")
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Average: {average_score:.1f}, Loss : {loss:.3f}")

if __name__ == "__main__":
    main()