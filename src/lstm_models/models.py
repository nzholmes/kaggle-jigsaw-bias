import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import SpatialDropout, Attention


class LstmGruAtten(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, lstm_units, maxlen):

        super(LstmGruAtten, self).__init__()
        embed_size = embedding_matrix.shape[1]
        self.maxlen = maxlen

        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.lstm_attention = Attention(lstm_units * 2, maxlen)

        dense_hidden_units = 6*lstm_units
        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.dropout = nn.Dropout(0.1)

        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        ##Attention Layer
        _, step_dim, _ = h_lstm2.shape
        if step_dim == self.maxlen:  # no padding
            h_lstm_atten2 = self.lstm_attention(h_lstm2)
        else:  # padding
            h_lstm_atten2 = self.lstm_attention(h_lstm2, step_dim=step_dim)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool, h_lstm_atten2), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out
