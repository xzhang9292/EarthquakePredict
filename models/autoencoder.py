import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv1d(1,2,4,stride=2),nn.ReLU(),
                                 nn.Conv1d(2,3,3,stride=2),nn.ReLU(),
                                 nn.Conv1d(3,3,3,stride=2),nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(1533,500),nn.Linear(500,100))
        self.decoder1 = nn.Sequential(nn.Linear(100,500),nn.Linear(500,1533))
        self.decoder2 = nn.Sequential(nn.ConvTranspose1d(3,3,3,stride=2),nn.ReLU(),
                                 nn.ConvTranspose1d(3,2,3,stride=2),nn.ReLU(),
                                 nn.ConvTranspose1d(2,1,4,stride=2),nn.Tanh())#,nn.Linear(),nn.ReLU(),

    def forward(self, data):
        scores = None
        out = self.encoder1(data.view(-1,1,4096))
        out = out.reshape(data.size(0),-1)
        out = self.encoder2(out)
        out = self.decoder1(out)
        out = out.reshape(data.size(0), 3, -1)
        scores = self.decoder2(out)
        scores = scores.reshape(-1,4096)
        return scores

