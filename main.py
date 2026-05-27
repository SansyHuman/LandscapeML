import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Example dataset with variable-length float sequences
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Custom collate function to pad sequences
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return padded, lengths, labels

# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        last_hidden = hidden[-1]  # shape: (batch, hidden_dim)
        return self.fc(last_hidden)

# Example data
seq1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float)   # length 2
seq2 = torch.tensor([[0.5, 0.6]], dtype=torch.float)               # length 1
seq3 = torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], dtype=torch.float)  # length 3

sequences = [seq1, seq2, seq3]
labels = [0, 1, 2]

dataset = SequenceDataset(sequences, labels)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

# Model setup
input_dim = 2
hidden_dim = 16
output_dim = 3
model = GRUModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3):
    for padded, lengths, labels in loader:
        optimizer.zero_grad()
        outputs = model(padded, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
