import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(37, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),  
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),  
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),  
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.value = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),        
        )
        self.cuda()

    def forward(self, x):
        x = self.net(x)
        value = self.value(x)
        return value
    