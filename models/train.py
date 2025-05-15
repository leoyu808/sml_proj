import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_trajectories(path, eta):
    df = pd.read_csv(path)
    state_cols = [c for c in df.columns if c.startswith("state_")]
    ind_cols = [c for c in df.columns if c.startswith("ind_")]

    trajectories, returns = [], []
    for _, grp in df.groupby("trajectory_id", sort=False):
        states = torch.tensor(grp[state_cols].values, dtype=torch.float32)
        excl = torch.tensor(grp["excl_action"].values, dtype=torch.long)
        ind = torch.tensor(grp[ind_cols].values, dtype=torch.float32)
        R = float(grp["return"].iloc[0])
        trajectories.append((states, excl, ind))
        returns.append(R)

    R_t = torch.tensor(returns, dtype=torch.float32)
    weights = torch.softmax(R_t / eta, dim=0)
    return trajectories, weights

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, weights):
        self.trajectories = trajectories
        self.weights      = weights
    def __len__(self):
        return len(self.trajectories)
    def __getitem__(self, idx):
        states, excl, ind = self.trajectories[idx]
        return states, excl, ind, self.weights[idx]

def collate_fn(batch):
    states, excl, ind, w = zip(*batch)
    seq_lens = [s.size(0) for s in states]
    T_max = max(seq_lens)
    B = len(states)
    D = states[0].size(1)
    K = ind[0].size(1)

    S = torch.zeros(B, T_max, D)
    E = torch.zeros(B, T_max, dtype=torch.long)
    I = torch.zeros(B, T_max, K)
    M = torch.zeros(B, T_max)
    for i, (s, e, ind_a) in enumerate(zip(states, excl, ind)):
        T = s.size(0)
        S[i, :T] = s
        E[i, :T] = e
        I[i, :T] = ind_a
        M[i, :T] = 1
    W = torch.tensor(w).unsqueeze(1)
    return S, E, I, W, M, seq_lens

class HybridActionRNNPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_ind, num_excl=2):
        super().__init__()
        self.rnn = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.excl_head = nn.Linear(hidden_dim, num_excl)
        self.ind_head  = nn.Linear(hidden_dim, num_ind)
    def forward(self, S, seq_lens):
        packed, _ = nn.utils.rnn.pack_padded_sequence(
            S, seq_lens, batch_first=True, enforce_sorted=False)
        out_p, _  = self.rnn(packed)
        out, _    = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        return out
    def logits(self, S, seq_lens):
        feat = self.forward(S, seq_lens)
        return self.excl_head(feat), self.ind_head(feat)

csv_path = "/scratch/network/ly4431/trajectories.csv"
eta = 1.0
trajectories, weights = load_trajectories(csv_path, eta)
dataset = TrajectoryDataset(trajectories, weights)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

state_dim = trajectories[0][0].size(1)
K_ind = trajectories[0][2].size(1)
hidden = 128

policy = HybridActionRNNPolicy(state_dim, hidden, num_ind=K_ind)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss(reduction='none')
bce_loss = nn.BCEWithLogitsLoss(reduction='none')

num_epochs = 20
for epoch in range(1, num_epochs+1):
    policy.train()
    total_loss = 0.0
    for S, E, I, W, M, seq_lens in loader:
        optimizer.zero_grad()
        excl_logits, ind_logits = policy.logits(S, seq_lens)
        B, T_max, _ = excl_logits.size()

        excl_flat = excl_logits.view(B*T_max, -1)
        E_flat = E.view(-1)
        loss_excl = ce_loss(excl_flat, E_flat).view(B, T_max)

        loss_ind = bce_loss(ind_logits, I).sum(-1)

        loss_t = loss_excl + loss_ind
        weighted = W * loss_t * M
        loss = weighted.sum() / (W * M).sum()

        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * B

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch:02d} — Loss: {avg_loss:.4f}")
