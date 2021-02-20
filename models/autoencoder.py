import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv1d(1,10,4,stride=2),nn.Tanh(),
                                 nn.Conv1d(10,20,3,stride=2),nn.Tanh(),
                                 nn.Conv1d(20,50,3,stride=2),nn.Tanh())
        self.encoder2 = nn.Sequential(nn.Linear(511*50,1000),nn.Tanh(),nn.Linear(1000,100),nn.Tanh())
        self.decoder1 = nn.Sequential(nn.Linear(100,1000),nn.Tanh(),nn.Linear(1000,511*50),nn.Tanh())
        self.decoder2 = nn.Sequential(nn.ConvTranspose1d(50,20,3,stride=2),nn.Tanh(),
                                 nn.ConvTranspose1d(20,10,3,stride=2),nn.Tanh(),
                                 nn.ConvTranspose1d(10,1,4,stride=2),nn.Tanh())#,nn.Linear(),nn.ReLU(),

    def forward(self, data):
        scores = None
        out = self.encoder1(data.view(-1,1,4096))
        out = out.reshape(data.size(0),-1)
        out = self.encoder2(out)
        out = self.decoder1(out)
        out = out.reshape(data.size(0), 50, -1)
        scores = self.decoder2(out)
        scores = scores.reshape(-1,4096)
        return scores

    def reduce(self, data):
        out = self.encoder1(data.view(-1,1,4096))
        out = out.reshape(data.size(0),-1)
        out = self.encoder2(out)
        return out
