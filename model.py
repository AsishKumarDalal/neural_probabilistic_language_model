import torch
import torch.nn as nn


class NeuralModel(nn.Module):
    def __init__(self, vocab_size, emb_size, cnxt_size, hid_state, hid_layers=1, d_conn=True, activation="relu", dropout=0.0):
        super().__init__()
        self.direct_connections = d_conn
        self.act = torch.relu if activation == "relu" else torch.tanh
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.embed_drop = nn.Dropout(dropout)
        self.hidden_drop = nn.Dropout(dropout)
        self.f_h_l = nn.Linear(cnxt_size * emb_size, hid_state)
        self.hidden = nn.ModuleList([nn.Linear(hid_state, hid_state) for _ in range(hid_layers - 1)])
        self.f_h_2 = nn.Linear(hid_state, vocab_size)
        if d_conn:
            self.d_nn = nn.Linear(cnxt_size * emb_size, vocab_size)

    def forward(self, x, logits=False):
        emb = self.embed_drop(self.embed(x)).view(x.size(0), -1)
        h = self.hidden_drop(self.act(self.f_h_l(emb)))

        for layer in self.hidden:
            h = self.hidden_drop(self.act(layer(h)))

        output = self.f_h_2(h)

        if self.direct_connections:
            output = output + self.d_nn(emb)

        return output if logits else torch.softmax(output, dim=1)
