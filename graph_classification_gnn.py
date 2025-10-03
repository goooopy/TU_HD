"""
train_gcn_nci1.py

Requirements:
  pip install torch dgl

Runs a simple GCN on the DGL TUDataset NCI1 for graph classification.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import TUDataset
from dgl.nn import GraphConv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# -------------------------
# Config / hyperparameters
# -------------------------
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
lr = 1e-3
num_epochs = 50
hidden_dim = 128
num_layers = 3
dropout = 0.5
add_self_loop = True
verbose_every = 1

torch.manual_seed(seed)
random.seed(seed)

# -------------------------
# Utility functions
# -------------------------
def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels).long()
    return batched_graph, labels

def set_node_features_onehot(dataset, node_label_key='node_labels'):
    """
    Ensure each graph has g.ndata['feat'] as a one-hot encoding of node labels
    If dataset graphs already have 'feat', this function will keep them.
    """
    # collect all labels to determine num classes
    max_label = -1
    for g, _ in dataset:
        # DGL TUDataset may store node labels under different keys,
        # we try node_label_key first then 'node_labels' or 'label' fallback.
        if node_label_key in g.ndata:
            labels = g.ndata[node_label_key]
        elif 'node_labels' in g.ndata:
            labels = g.ndata['node_labels']
        elif 'label' in g.ndata:
            labels = g.ndata['label']
        else:
            labels = None

        if labels is not None:
            max_label = max(max_label, int(labels.max().item()))

    if max_label < 0:
        # No integer node labels found. We'll assume node features already exist.
        print("No node label integers found in dataset; leaving g.ndata['feat'] as-is (if present).")
        return

    num_classes = max_label + 1
    print(f"Detected node label classes: {num_classes}")

    # convert to one-hot and store as 'feat'
    for g, _ in dataset:
        if 'feat' in g.ndata:
            # keep existing continuous features
            continue

        if node_label_key in g.ndata:
            labels = g.ndata[node_label_key].long()
        elif 'node_labels' in g.ndata:
            labels = g.ndata['node_labels'].long()
        elif 'label' in g.ndata:
            labels = g.ndata['label'].long()
        else:
            labels = None

        if labels is None:
            continue

        onehot = F.one_hot(labels, num_classes=num_classes).float()
        g.ndata['feat'] = onehot.squeeze()
    print('g feat', onehot.size())

# -------------------------
# Model
# -------------------------
class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_layers=3, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first layer
        self.convs.append(GraphConv(in_dim, hidden_dim, norm='both', weight=True, bias=True))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # middle layers
        for _ in range(n_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, g, x):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(g, h)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        # global pooling: mean over nodes per graph
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')   # shape (batch_size, hidden_dim)
        out = self.classify(hg)
        return out

# -------------------------
# Load dataset and preprocess
# -------------------------
print("Loading NCI1 dataset...")
dataset = TUDataset('NCI1')   # downloads if needed

# ensure node features exist under 'feat'
set_node_features_onehot(dataset, node_label_key='node_labels')

# Add self-loops (optional)
if add_self_loop:
    print("Adding self-loops to graphs (if missing)...")
    # we will inplace modify each graph returned by the dataset by replacing it with a new graph containing self-loops
    processed_graphs = []
    processed_labels = []
    for g, y in dataset:
        # dgl.add_self_loop returns a new graph with self-loops added (keeps existing edge/ndata/edata)
        g2 = dgl.add_self_loop(g)
        # preserve features: add_self_loop already copies ndata/edata but if it doesn't, ensure it's set
        if 'feat' not in g2.ndata and 'feat' in g.ndata:
            g2.ndata['feat'] = g.ndata['feat']
        processed_graphs.append(g2)
        processed_labels.append(y)
    # replace dataset graphs with the processed ones by creating a simple list dataset
    dataset_graphs = processed_graphs
    dataset_labels = processed_labels
else:
    dataset_graphs = [g for g, _ in dataset]
    dataset_labels = [y for _, y in dataset]

# Convert labels to ints
dataset_labels = [int(y) for y in dataset_labels]
num_graphs = len(dataset_graphs)
print(f"Number of graphs: {num_graphs}")

# Build train/val/test split (stratify by label)
train_idx, test_idx = train_test_split(range(num_graphs), test_size=0.2, stratify=dataset_labels, random_state=seed)
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=[dataset_labels[i] for i in train_idx], random_state=seed)
# this gives ~70% train, 10% val, 20% test

def subset(dataset_graphs, dataset_labels, indices):
    return [(dataset_graphs[i], dataset_labels[i]) for i in indices]

train_set = subset(dataset_graphs, dataset_labels, train_idx)
val_set = subset(dataset_graphs, dataset_labels, val_idx)
test_set = subset(dataset_graphs, dataset_labels, test_idx)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# infer input dim and number of classes
sample_g, sample_label = dataset_graphs[0], dataset_labels[0]
if 'feat' in sample_g.ndata:
    in_dim = sample_g.ndata['feat'].shape[-1]
else:
    # fallback: use degree one-hot up to some max degree
    print("No g.ndata['feat'] found — using degree one-hot encoding up to max degree 10.")
    max_deg = 10
    for g in dataset_graphs:
        deg = g.in_degrees().clamp(max=max_deg)
        g.ndata['feat'] = F.one_hot(deg, num_classes=max_deg+1).float()
    in_dim = max_deg + 1

n_classes = len(set(dataset_labels))
print(f"in_dim={in_dim}, classes={n_classes}")

# -------------------------
# Model / optimizer / training
# -------------------------
model = GCNClassifier(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=n_classes,
                      n_layers=num_layers, dropout=dropout).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

def evaluate(loader):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for batched_graph, labels in loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feats = batched_graph.ndata['feat'].to(device)
            logits = model(batched_graph, feats)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# training loop
best_val_acc = 0.0
best_test_acc = 0.0
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batched_graph, labels in train_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feats = batched_graph.ndata['feat'].to(device)

        logits = model(batched_graph, feats)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * labels.size(0)

    train_loss = epoch_loss / len(train_set)
    val_loss, val_acc = evaluate(val_loader)
    test_loss, test_acc = evaluate(test_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    if epoch % verbose_every == 0:
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f} | test_acc {test_acc:.4f}")

print("Training finished.")
print(f"Best validation acc: {best_val_acc:.4f}; corresponding test acc: {best_test_acc:.4f}")

