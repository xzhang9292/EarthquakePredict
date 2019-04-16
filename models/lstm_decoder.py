import torch.nn as nn
import torch.nn.init as init
class Decoder(nn.Module):
	def __init__(self, hid_dim, output, n_layers):
		super().__init__()
		self.lstm = nn.LSTM(hid_dim, output, n_layers)
	def forward(self, src):
		outputs, (hidden, cell) = self.lstm(src)
		return output, hidden, cell