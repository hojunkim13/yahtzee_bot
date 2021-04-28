import torch
from Network import Network
from Simulator import *
from MCTS import MCTS
from collections import deque
import random

class Agent:
    def __init__(self, lr, batch_size, maxlen, n_sim) -> None:
        self.net = Network()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = lr)
        self.batch_size = batch_size
        self.mcts = MCTS(self.net)
        self.memory = deque(maxlen = maxlen)
        self.outcome_memory = deque(maxlen = maxlen)
        self.n_sim = n_sim
        

    def getAction(self):
        action = self.mcts.search(self.n_sim)        
        self.step_count += 1
        return action
    
    
    def pushMemory(self, tmp_memory, outcome):
        outcome = deque([outcome] * len(tmp_memory))
        self.outcome_memory += outcome
        self.memory += tmp_memory
                

    def learn(self):
        idx_max = len(self.memory)        
        indice = random.sample(range(idx_max), min(self.batch_size, idx_max))
        memory = np.array(self.memory, dtype = np.float32)[indice]
        outcome = np.array(self.outcome_memory, dtype = np.float32)[indice]

        S = torch.tensor(memory, dtype = torch.float).cuda().reshape(-1, 22, 6)                
        _, value = self.net(S)
        outcome = torch.tensor(outcome, dtype = torch.float).cuda().view(*value.shape)
        value_loss = torch.square(value - outcome).mean()
        
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()        
        return value_loss.item()

    def save(self, env_name):
        torch.save(self.net.state_dict(), f"./model/{env_name}.pt")

    def load(self, env_name):
        state_dict = torch.load(f"./model/{env_name}.pt")
        self.net.load_state_dict(state_dict)