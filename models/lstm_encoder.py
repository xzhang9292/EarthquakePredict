import torch.nn as nn
import torch.nn.init as init

class Encoder(nn.Module):
	def __init__(self, input_dim, hid_dim, n_layers):
		super().__init__()
		self.lstm = nn.LSTM(inputdim, hid_dim, n_layers)
	def forward(self, src):
		outputs, (hidden, cell) = self.lstm(src)
		return output, hidden, cell