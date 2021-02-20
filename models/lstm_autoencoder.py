import torch
import models.lstm_encoder as lstm_encoder
import models.lstm_decoder as lstm_decoder
from torch import nn
class Seq2seq(nn.Module):

    def __init__(self, inputdim, hidden, layers):
        super(Seq2seq, self).__init__()
        self.n_layers = layers
        self.hidden_size = hidden
        self.encoder = lstm_encoder.Encoder(inputdim, hidden, self.n_layers)
        self.decoder = lstm_decoder.Decoder(hidden, inputdim, self.n_layers)
    def forward(self, src):
        batch, seq_size = src.shape
        output,hidden, cell = self.encoder(src)
        outputt = torch.zeros([seq_size,batch,self.hidden_size])
        for i in torch.arange(0,seq_size,1):
            outputt[i,:,:] = hidden[0,:,:]
        output2 = self.decoder(outputt)
        return torch.transpose(output2,0,1).reshape(batch,seq_size)