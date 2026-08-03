import torch
import torch.nn as nn


class NeuralModel(nn.Module):
    def __init__(self, vocab_size, emb_size, cnxt_size, hid_state, d_conn=True):
        super().__init__()
        self.direct_connections = d_conn
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.f_h_l = nn.Linear(cnxt_size * emb_size, hid_state)
        self.f_h_2 = nn.Linear(hid_state, vocab_size)
        if d_conn:
            self.d_nn = nn.Linear(cnxt_size * emb_size, vocab_size)

    def forward(self, x, logits=False):
        emb = self.embed(x).view(x.size(0), -1)
        hidden = torch.tanh(self.f_h_l(emb))
        output = self.f_h_2(hidden)

        if self.direct_connections:
            output = output + self.d_nn(emb)

        return output if logits else torch.softmax(output, dim=1)
