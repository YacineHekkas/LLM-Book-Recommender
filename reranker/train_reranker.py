import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PAIRS_IN = "train_pairs.npz"
MODEL_OUT = "reranker.pt"

class Reranker(nn.Module):
    def __init__(self, q_dim, d_dim, hidden_dim=64):
        super().__init__()
        # map query & doc embeddings into same hidden space
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.d_proj = nn.Linear(d_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, q, d):
        q = self.q_proj(q)
        d = self.d_proj(d)
        x = torch.cat([q, d], dim=1)
        return torch.sigmoid(self.fc(x))

def train():
    data = np.load(PAIRS_IN, allow_pickle=True)
    q_embs = torch.tensor(data["q_embs"], dtype=torch.float32)
    d_embs = torch.tensor(data["d_embs"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.float32).unsqueeze(1)

    q_dim = q_embs.shape[1]
    d_dim = d_embs.shape[1]
    model = Reranker(q_dim, d_dim)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(q_embs, d_embs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"✅ Saved model to {MODEL_OUT}")

if __name__ == "__main__":
    train()
