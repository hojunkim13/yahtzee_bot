import numpy as np
from Env import Yahtzee
from Agent import Agent
from collections import deque
from Simulator import *

lr = 1e-3
batch_size = 256
n_sim = 5000

maxlen = 50000
n_episode = 10000
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
            action = agent.getAction(state)
            tmp_memory.append(preprocessing(state))        
            state, reward, done, _ = env.step(action)    
            score += reward            
        outcome = calcOutcome(state)
        agent.pushMemory(tmp_memory, outcome)
        
        
        #if (e+1) % 1 == 0:
        loss = agent.learn()
        agent.save("Yahtzee")
        # else:
        #     loss = 0
        score_list.append(score)
        average_score = np.mean(score_list[-100:])        
        print(f"Episode : {e+1} / {n_episode}, Score : {score}, Average: {average_score:.1f}, Loss : {loss:.3f}")

if __name__ == "__main__":
    main()