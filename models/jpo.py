import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HybridActionRNNPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_action, num_path):
        super().__init__()
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.path_head   = nn.Linear(hidden_dim, num_path)
        self.action_head = nn.Linear(hidden_dim, num_action)

    def forward(self, states, seq_lens):
        packed, _  = nn.utils.rnn.pack_padded_sequence(
                        states, seq_lens, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out

    def logits(self, states, seq_lens):
        hidden = self.forward(states, seq_lens)
        path_logits   = self.path_head(hidden)
        action_logits = self.action_head(hidden)
        return path_logits, action_logits

