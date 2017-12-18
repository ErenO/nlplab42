import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torch.nn as nn
from torch.autograd import Variable

class BowModel(nn.Module):
    def __init__(self, emb_tensor):
        super(BowModel, self).__init__()
        n_embedding, dim = emb_tensor.size()
        self.embedding = nn.Embedding(n_embedding, dim, padding_idx=0)
        self.embedding = nn.Embedding(n_embedding, dim, padding_idx=0)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.embedding.weight = Parameter(emb_tensor, requires_grad=False)
        self.out = nn.Linear(dim, 2)

    def forward(self, input):
        '''
        input is a [batch_size, sentence_length] tensor with a list of token IDs
        '''
        embedded = self.embedding(input)
        # Here we take into account only the first word of the sentence
        # You should change it, e.g. by taking the average of the words of the sentence
        bow = embedded[:, 0]
        bow = embedded.mean(dim=1)
        lstm_out, self.hidden = self.lstm(
            embeds.view(input[1], 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(input[1], -1))
        # tag_scores = F.log_softmax(tag_space)
        return F.log_softmax(tag_space)
