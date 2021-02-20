import torch
import torch.nn as nn
import torch.nn.init as init
class Decoder(nn.Module):
	def __init__(self, inputdim,hid_dim, layers):
		super(Decoder,self).__init__()
		self.hidden = hid_dim
		self.n_layers = layers
		self.lstm = nn.LSTM(inputdim, hid_dim, layers)
	def forward(self,output):
		seqsize,batch,hiddensize = output.shape
		h0 = torch.randn(self.n_layers,batch,self.hidden)
		c0 = torch.randn(self.n_layers,batch,self.hidden)
		outputs, (hidden, cell) = self.lstm(output,(h0, c0))
		return outputs