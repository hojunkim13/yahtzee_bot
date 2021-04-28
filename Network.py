import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(22, 64, 1, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.policy = nn.Sequential(
            nn.Linear(64 * 6 , 256),
            nn.ReLU(),
            nn.Linear(256, 44),
            nn.Softmax(-1)
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 6 , 256),
            nn.ReLU(),
            nn.Linear(256, 1),        
        )
        self.cuda()

    def forward(self, x):
        x = self.net(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value