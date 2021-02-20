import torch
import torch.nn as nn
import torch.nn.init as init

class Encoder(nn.Module):
	def __init__(self, input_dim, hid_dim,layers):
		super(Encoder,self).__init__()
		self.n_layers = layers
		self.hidden = hid_dim
		self.inputd = input_dim
		self.lstm = nn.LSTM(input_dim, hid_dim, layers)
	def forward(self, src):
		batch, seqsize = src.shape
		datainput = torch.zeros([seqsize,batch,self.inputd])
		datainput[:,:,0] = torch.transpose(src,0,1)[:,:]
		h0 = torch.randn(self.n_layers,batch,self.hidden)
		c0 = torch.randn(self.n_layers,batch,self.hidden)
		outputs, (hidden, cell) = self.lstm(datainput,(h0,c0))
		return torch.transpose(outputs,0,1), hidden, cell