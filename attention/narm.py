# This code was originally an implementation of Li, Jing, et al.
# It originated from:
# https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NARM(nn.Module):
    """Neural Attentive Session Based Recommendation Model.

    Note that this module performs its own embedding on the input.
    This may be unnecessary, as we're already getting some high quality BERT
    embeddings. TODO: Check this with Michael and Hamlin.

    Args:
        n_items(int):       The number of items we'll be receiving as input
                            (ie. the vocabulary size).
        hidden_size(int):   The width of a GRU hidden layer.
        embedding_dim(int): The dimension of our item embedding.
        batch_size(int):    The batch size for our network training.
        n_layers(int):      The number of GRU layers to use.
    """
    def __init__(self, n_items, hidden_size, embedding_dim, batch_size, n_layers=1):
        super(NARM, self).__init__()
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers

        # TODO: we embed the input again here. Is this necessary if we already
        # have our BERT module set up before this?
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.25)

        # GRU + Attention layer
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)

        # Final feedforward fully-connected layer.
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        # self.sf = nn.Softmax()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, seq, lengths):
        """Predict (unnormalized) item scores from an input sequence.

        Args:
            Seq: An input sequence of items to predict. These items should be
                 given in ItemID form -- embeddings are module-internal as of
                 now.
        """
        hidden = self.init_hidden(seq.size(1))
        embs = self.emb_dropout(self.emb(seq))
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)

        # Fetch the last hidden state of the last timestamp.
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device=self.device),
                torch.tensor([0.], device=self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size))
        alpha = alpha.view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        item_embs = self.emb(torch.arange(self.n_items).to(self.device))

        # Final fully-connected linear layer.
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))
        # scores = self.sf(scores)

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size),
                requires_grad=True).to(self.device)
