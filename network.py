# RNN-LSTM循环神经网络

import torch.nn as nn


class ThreeLayerLSTM(nn.Module):
    def __init__(self, num_pitch, ninp, nhid, nlayers, dropout):
        super(ThreeLayerLSTM, self).__init__()

        # define the architecture
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_pitch, ninp)
        self.LSTM = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, num_pitch)
        self.init_weights() 
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.LSTM(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0) * output.size(1), -1))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weights = next(self.parameters())
        return (weights.new_zeros((self.nlayers, batch_size, self.nhid), requires_grad=requires_grad),
                weights.new_zeros((self.nlayers, batch_size, self.nhid), requires_grad=requires_grad))


class LSTM_BiDir(nn.Module):
    def __init__(self, num_pitch, ninp, nhid, nlayers, dropout):
        super(LSTM_BiDir, self).__init__()

        # define the architecture
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(num_pitch, ninp)
        # LSTM 的输出为 (seq_len, batch_size, hidden_size*num_directions)
        self.LSTM = nn.LSTM(input_size=ninp, hidden_size=nhid, num_layers=nlayers, bidirectional=True, dropout=dropout, batch_first=True)
        # Linear 是将 nhid*2 的输入 转化为 num_pitch的输出(类似于分类)
        self.decoder = nn.Linear(nhid * 2, num_pitch)
        self.init_weights() 
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        # 构建词嵌入
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.LSTM(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0) * output.size(1), -1))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, batch_size, requires_grad=True):
        weights = next(self.parameters())
        return (weights.new_zeros((self.nlayers * 2, batch_size, self.nhid), requires_grad=requires_grad),
                weights.new_zeros((self.nlayers * 2, batch_size, self.nhid), requires_grad=requires_grad))

