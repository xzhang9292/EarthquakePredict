import models.lstm_encoder
import models.lstm_decoder
from torch import nn
class Seq2seq(nn.Module):

    def __init__(self, inputdim, hidden, n_layers):
        super(Seq2seq, self).__init__()
        self.n_layers = layers
        self.encoder = lstm_encoder.Encoder(inputdim, hidden, self.n_layers)
        self.decoder = lstm_decoder.Decoder(hidden, inputdim, self.n_layers)
    def forward(self, src):
     	inputsize = src.shape[0]
     	output,hidden, cell = self.encoder(src)
     	output2,hidden2,cell2 = self.decoder(output2)
     	return output2