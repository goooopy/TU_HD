"""GCN using basic message passing
here we directly bundle the messages without having to compute anything. Thus all intermediate layers have in_features size
"""
import argparse, time, math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
#from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset
#from geom_gcn import ChameleonDataset, CornellDataset, TexasDataset, WisconsinDataset
from torch.utils.data import TensorDataset, DataLoader
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from dgl.data import TUDataset
from torch.utils.data import DataLoader, random_split
import copy
import dgl.function as fn
from stack_ensemble import stack_ensemble

# 2. Collate function for DataLoader
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


class EdgeGNNLayer(nn.Module):
    def __init__(self, aggr='mean'):
        super().__init__()
        self.aggr = aggr

    def forward(self, g, edge_feats):
        """
        g: DGLGraph
        edge_feats: [num_edges, in_dim]
        """
        # Build line graph (edges of g become nodes in lg)
        lg = dgl.line_graph(g, backtracking=False)
        lg.ndata['h'] = edge_feats 

        # Edge-to-edge message passing
        if self.aggr == 'mean':
            lg.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_new'))
        elif self.aggr == 'sum':
            lg.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_new'))
        elif self.aggr == 'max':
            lg.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h_new'))
        else:
            raise ValueError(f"Unknown aggregation {self.aggr}")

        # Apply transformation
        updated_edge_feats = lg.ndata['h_new']

        updated_edge_feats = args.alpha * edge_feats + (1-args.alpha) * updated_edge_feats
        return updated_edge_feats


class EdgeGNN(nn.Module):
    def __init__(self, num_layers=2, aggr='sum'):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        
        # subsequent edge-only layers
        for _ in range(num_layers):
            self.layers.append(EdgeGNNLayer(aggr=aggr))

    def forward(self, g, edge_feats):
        """
        g: DGLGraph
        node_feats: [num_nodes, node_dim]
        """
        # build initial edge features as concatenation of endpoints
        #g.ndata['h'] = node_features
        #g.apply_edges(lambda edges: {
        #    'ef': torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
        #})
        #edge_feats = g.edata['ef']

        # propagate through edge layers
        for layer in self.layers:
            edge_feats = layer(g, edge_feats)
        g.edata['ef'] = edge_feats
        eg = dgl.mean_edges(g, 'ef')

        return eg 

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


from sklearn.cluster import KMeans

def classify_by_multi_prototypes(logits, labels, train_mask, test_mask, n_class, prototypes_per_class=3, use_cosine=True):
    device = logits.device
    train_logits = logits[train_mask].detach().cpu()
    train_labels = labels[train_mask].detach().cpu()
    test_logits = logits[test_mask].detach().cpu()
    true_labels = labels[test_mask].detach().cpu()

    all_prototypes = []
    proto_labels = []

    # Generate multiple prototypes for each class
    for c in range(n_class):
        class_mask = (train_labels == c)
        class_samples = train_logits[class_mask]  # [N_c, D]

        if class_samples.size(0) < prototypes_per_class:
            # Not enough samples, fallback to single mean
            proto = class_samples.mean(dim=0, keepdim=True)
            all_prototypes.append(proto)
            proto_labels.extend([c])
        else:
            # KMeans clustering
            kmeans = KMeans(n_clusters=prototypes_per_class, n_init='auto', random_state=42)
            kmeans.fit(class_samples)
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            all_prototypes.append(centroids)
            proto_labels.extend([c] * prototypes_per_class)

    # Stack all prototypes [num_total_prototypes, D]
    all_prototypes = torch.cat(all_prototypes, dim=0)
    proto_labels = torch.tensor(proto_labels)

    if use_cosine:
        all_prototypes = F.normalize(all_prototypes, dim=1)
        test_logits = F.normalize(test_logits, dim=1)
        similarity = torch.matmul(test_logits, all_prototypes.T)  # [N_test, N_prototypes]
        pred_indices = similarity.argmax(dim=1)
    else:
        # Euclidean distance
        dists = torch.cdist(test_logits, all_prototypes)  # [N_test, N_prototypes]
        pred_indices = dists.argmin(dim=1)

    pred_labels = proto_labels[pred_indices]

    # Compute accuracy
    correct = (pred_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total

    return pred_labels, accuracy

def classify_by_weighted_prototype(logits, labels, train_mask, test_mask, n_class, temperature=1):
    # Extract training and test logits
    train_logits = logits[train_mask]
    train_labels = labels[train_mask]
    test_logits = logits[test_mask]
    true_labels = labels[test_mask]

    # Placeholder for refined class means
    refined_means = []

    for c in range(n_class):
        class_mask = (train_labels == c)
        class_samples = train_logits[class_mask]  # [N_c, D]

        if class_samples.size(0) == 0:
            # No samples for this class — use zeros
            refined_means.append(torch.zeros_like(train_logits[0]))
            continue

        # Step 1: compute initial mean
        class_mean = class_samples.mean(dim=0, keepdim=True)  # [1, D]

        # Step 2: compute cosine similarity of each point to the class mean
        class_samples_norm = F.normalize(class_samples, dim=1)
        class_mean_norm = F.normalize(class_mean, dim=1)
        sims = torch.matmul(class_samples_norm, class_mean_norm.T).squeeze(1)  # [N_c]

        # Step 3: use softmax over similarity as weights
        weights = F.softmax(sims / temperature, dim=0)  # [N_c]

        # Step 4: weighted mean
        weighted_mean = (weights.unsqueeze(1) * class_samples).sum(dim=0)
        refined_means.append(weighted_mean)

    # Stack refined means: [n_class, D]
    refined_means = torch.stack(refined_means)
    refined_means = F.normalize(refined_means, dim=1)
    test_logits = F.normalize(test_logits, dim=1)

    # Cosine similarity to each class mean
    similarity = torch.matmul(test_logits, refined_means.T)  # [N_test, n_class]
    pred_labels = similarity.argmax(dim=1)

    # Accuracy
    correct = (pred_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total

    return pred_labels, accuracy

def classify_by_nearest_neighbor(logits, labels, train_mask, test_mask,n_class):
    # Extract training and test logits
    train_logits = logits[train_mask]          # [N_train, D]
    test_logits = logits[test_mask]            # [N_test, D]
    train_labels = labels[train_mask]          # [N_train]

    # Normalize for cosine similarity
    train_logits = F.normalize(train_logits, dim=1)  # [N_train, D]
    test_logits = F.normalize(test_logits, dim=1)    # [N_test, D]

    # Compute cosine similarity between test and all training samples
    similarity = torch.matmul(test_logits, train_logits.T)  # [N_test, N_train]

    # For each test sample, find the most similar training sample
    nearest_indices = similarity.argmax(dim=1)  # [N_test]
    pred_labels = train_labels[nearest_indices]  # [N_test]

    # Get ground-truth labels for test samples
    true_labels = labels[test_mask]

    # Compute accuracy
    correct = (pred_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total

    return pred_labels, accuracy


def batchwise_greedy_transductive_classification(logits, labels, train_mask, test_mask, n_class, step_size=50):
    logits = logits.detach().clone()
    labels = labels.clone()

    # Normalize all logits
    logits = F.normalize(logits, dim=1)

    # Indices
    train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()
    test_idx = test_mask.nonzero(as_tuple=True)[0].tolist()
    remaining = set(test_idx)

    # Track pseudo-labels
    pseudo_labels = labels.clone()
    known_indices = set(train_idx)

    # Initial classification using class means
    def compute_class_means(indices):
        class_means = []
        for c in range(n_class):
            class_samples = [i for i in indices if pseudo_labels[i] == c]
            if class_samples:
                mean = logits[class_samples].mean(dim=0)
            else:
                mean = torch.zeros_like(logits[0])
            class_means.append(mean)
        class_means = torch.stack(class_means)
        return F.normalize(class_means, dim=1)

    while remaining:
        class_means = compute_class_means(known_indices)

        # Step 1: classify all remaining test samples
        remaining_indices = list(remaining)
        sims = torch.matmul(logits[remaining_indices], class_means.T)  # [N_remaining, n_class]
        max_sims, pred_classes = sims.max(dim=1)

        # Step 2: select top-k (step_size) most confident samples
        sorted_indices = torch.argsort(max_sims, descending=True)
        selected = sorted_indices[:step_size].tolist()

        for idx in selected:
            real_idx = remaining_indices[idx]
            pseudo_labels[real_idx] = pred_classes[idx].item()
            known_indices.add(real_idx)
            remaining.remove(real_idx)

    # Final test predictions and accuracy
    pred_labels = pseudo_labels[test_mask]
    true_labels = labels[test_mask]
    acc = (pred_labels == true_labels).float().mean().item()

    return pred_labels, acc

def classify_by_similarity(logits, labels, train_mask, test_mask, n_class):
    # Compute mean logit vector for each class using training samples
    class_means = []
    for c in range(n_class):
        class_mask = (labels == c) & train_mask
        if class_mask.sum() == 0:
            class_means.append(torch.zeros_like(logits[0]))
        else:
            class_mean = logits[class_mask].mean(dim=0)
            class_means.append(class_mean)
    class_means = torch.stack(class_means)  # [n_class, D]

    # Normalize for cosine similarity
    class_means = F.normalize(class_means, dim=1)
    test_logits = F.normalize(logits[test_mask], dim=1)

    # Cosine similarity between test samples and class means
    similarity = torch.matmul(test_logits, class_means.T)  # [N_test, n_class]

    # Predict labels based on maximum similarity
    pred_labels = similarity.argmax(dim=1)

    # Ground truth labels for test samples
    true_labels = labels[test_mask]

    # Compute accuracy
    correct = (pred_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total

    return pred_labels, accuracy


def greedy_transductive_classification(logits, labels, train_mask, test_mask, n_class):
    # Clone and prepare
    logits = logits.detach().clone()
    labels = labels.clone()
    train_idx = train_mask.nonzero(as_tuple=True)[0].tolist()
    test_idx = test_mask.nonzero(as_tuple=True)[0].tolist()

    # Maintain known labels: start with training
    known_indices = set(train_idx)
    pseudo_labels = labels.clone()
    remaining = set(test_idx)

    # Normalize once for cosine similarity
    logits = F.normalize(logits, dim=1)

    while remaining:
        # Step 1: compute current class means from known samples
        class_means = []
        for c in range(n_class):
            class_indices = [i for i in known_indices if pseudo_labels[i] == c]
            if class_indices:
                mean_vec = logits[class_indices].mean(dim=0)
            else:
                mean_vec = torch.zeros_like(logits[0])
            class_means.append(mean_vec)
        class_means = torch.stack(class_means)  # [n_class, D]
        class_means = F.normalize(class_means, dim=1)

        # Step 2: evaluate similarity of remaining test samples to means
        best_score = -float("inf")
        best_idx = None
        best_label = None

        for idx in remaining:
            sim = torch.matmul(logits[idx], class_means.T)  # [n_class]
            max_sim, pred_label = sim.max(dim=0)

            if max_sim > best_score:
                best_score = max_sim
                best_idx = idx
                best_label = pred_label.item()

        # Step 3: assign label and update known set
        pseudo_labels[best_idx] = best_label
        known_indices.add(best_idx)
        remaining.remove(best_idx)

    # Final accuracy on original test set
    pred_labels = pseudo_labels[test_mask]
    true_labels = labels[test_mask]
    acc = (pred_labels == true_labels).float().mean().item()

    return pred_labels, acc

def main(args):
    # load and preprocess dataset
    if args.dataset == 'mutag':
        dataset = TUDataset(name='MUTAG')
    elif args.dataset == 'enzymes':
        dataset = TUDataset(name='ENZYMES')
    elif args.dataset == 'proteins':
        dataset = TUDataset(name='PROTEINS')
    elif args.dataset == 'nci1':
        dataset = TUDataset(name='NCI1')
    elif args.dataset == 'dd':
        dataset = TUDataset(name='DD')
    elif args.dataset == 'ptc_fm':
        dataset = TUDataset(name='PTC_FM')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        print('using CUDA')
    print('dataset', dataset)
    n_classes = dataset.num_labels
    print('num labels', n_classes)
    g, label = dataset[0]
    print('inspecting the first graph', g, 'node labels:\n', g.ndata['node_labels'])
    print('e data:', g.edata)
    if 'node_attr' in g.ndata:
        print('node attributes', g.ndata['node_attr'])

    num_graphs = len(dataset)
    num_train = int(0.8 * num_graphs)
    num_val   = int(0.1 * num_graphs)
    #num_val   = 1
    num_test  = num_graphs - num_train - num_val  # ensure all graphs used

    print('SPLIT=====', num_train, num_val, num_test)

    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(args.seed)  # reproducibility
        )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, collate_fn=collate) # note shuffle is False
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate)


    # we want to create edge fetures

    if args.use_feature == 'node_attr': 
        assert('node_attr' in g.ndata)
        if g.ndata['node_attr'].size(-1) == 1: # the node attribute is only one number
            max_node_l = 0
            min_node_l = 100000 
            for g, label in dataset:
                l = g.ndata['node_attr']
                max_node_l = max(max_node_l, max(l))
                min_node_l = min(min_node_l, min(l))
            max_node_l = int(max_node_l.item() + 1 )
            min_node_l = int(min_node_l.item())
            print('max_node_l', max_node_l, 'min_node_l', min_node_l)
            for g, label in dataset:
                x = F.one_hot(g.ndata['node_attr'].long()-min_node_l, num_classes= (max_node_l-min_node_l)).float()
                g.ndata['feat'] = x
        else:
            for g, label in dataset:
                x = g.ndata['node_attr'].float()
                g.ndata['feat'] = x
        in_feats = x.size(-1) 
        print('using node_attr', x )
    elif args.use_feature == 'node_labels':
        print('using node_labels')
        assert('node_labels' in g.ndata)
        max_node_l = 0
        for g, label in dataset:
            max_node_l = max(max_node_l, max(g.ndata['node_labels']))
        max_node_l = max_node_l.item() + 1
        print('max_node_l', max_node_l)
        for g, label in dataset:
            x = F.one_hot(g.ndata['node_labels'], num_classes=max_node_l).float()
            g.ndata['feat'] = x
        in_feats = x.size(-1) 
        print('in_feats', in_feats)

    # edge_only means we do not use node features. otherwise we add the two features of the end point into one (part of) edge feature
    if args.use_feature != 'edge_only': 
        for g, label in dataset:
            g.apply_edges(lambda edges: {
            'e': edges.src['feat']+ edges.dst['feat'] }) # use concatenation of end point features as e feature
            #'e': torch.cat([edges.src['feat'], edges.dst['feat']], dim=-1) }) # use concatenation of end point features as e feature

    if 'edge_labels' in g.edata:
        max_edge_l = 0
        for g, label in dataset:
            max_edge_l = max(max_edge_l, max(g.edata['edge_labels']))
        max_edge_l = max_edge_l.item() + 1
        print('max_edge_l', max_edge_l)
        for g, label in dataset:
            y = F.one_hot(g.edata['edge_labels'], num_classes=max_edge_l).float()
            if args.use_feature == 'edge_only':
                g.edata['feat'] = y
            else:
                #g.edata['feat'] = torch.cat((g.edata['e'],y), dim=-1)
                g.edata['feat'] = g.edata['e']

    print('g  edge feat', g.edata['feat'].size())

    model = EdgeGNN(num_layers=args.n_layers, aggr='mean')

    if cuda:
        model.cuda()

    # bundling

    all_features = []
    all_labels = []
    t0 = time.time()
    with torch.no_grad():
        for batched_graph, labels in train_loader:
            if cuda:
                batched_graph = batched_graph.to(args.gpu)
                features = batched_graph.edata['feat'].to(args.gpu)
            logits = model(batched_graph, features )
            #print('logits', logits)
            all_features.append(logits)
            all_labels.append(labels)
        t1 = time.time()
        print('training time', t1-t0)

        t0 = time.time()
        for batched_graph, labels in val_loader:
            if cuda:
                batched_graph = batched_graph.to(args.gpu)
                features = batched_graph.edata['feat'].to(args.gpu)
            logits = model(batched_graph, features)
            all_features.append(logits)
            all_labels.append(labels)
        t1 = time.time()
        print('val time', t1-t0)

        t0 = time.time()
        for batched_graph, labels in test_loader:
            if cuda:
                batched_graph = batched_graph.to(args.gpu)
                features = batched_graph.edata['feat'].to(args.gpu)
            logits = model(batched_graph, features )
            all_features.append(logits)
            all_labels.append(labels)
        t1 = time.time()
        print('test time', t1-t0)

    all_features = torch.cat(all_features, dim=0).squeeze()
    print('all_feature size', all_features.size())

    all_labels = torch.cat(all_labels, dim=0).squeeze()
    print('all_labels size', all_labels.size())
    # lets make train_mask, val_mask, and test_mask so that we can use classify_by_similarity that was written for node 
    # classification. 
    # we simply create a one_D tensor with the first part being the the train samples. Note train, val, and test are
    # concatenated into all_features
    train_mask = torch.zeros(num_graphs, dtype=torch.bool)
    val_mask   = torch.zeros(num_graphs, dtype=torch.bool)
    test_mask  = torch.zeros(num_graphs, dtype=torch.bool)

    train_mask[:num_train] = True
    val_mask[num_train:num_train+num_val] = True
    test_mask[num_train+num_val:] = True

    logits = all_features
    labels = all_labels
    if cuda:
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    pred_labels, acc = classify_by_similarity(logits, labels, train_mask, val_mask, n_classes)
    print(" classify_by_similarity Val Accuracy {:.4f}".format(acc))
    pred_labels, acc = classify_by_similarity(logits, labels, train_mask, test_mask, n_classes)
    print("Test Accuracy {:.4f}".format(acc))


    pred_labels, acc = classify_by_nearest_neighbor(logits, labels, train_mask, val_mask, n_classes)
    print("classify_by_nearest_neighbor Val Accuracy {:.4f}".format(acc))
    pred_labels, acc = classify_by_nearest_neighbor(logits, labels, train_mask, test_mask, n_classes)
    print("classify_by_nearest_neighbor Val Accuracy Test Accuracy {:.4f}".format(acc))

    pred_labels, acc =  classify_by_weighted_prototype(logits, labels, train_mask, val_mask, n_classes)
    print("classify_by_weighted_prototype Val Accuracy {:.4f}".format(acc))
    pred_labels, acc =  classify_by_weighted_prototype(logits, labels, train_mask, test_mask, n_classes)
    print("Test Accuracy {:.4f}".format(acc))

    pred_labels, acc =classify_by_multi_prototypes(logits, labels, train_mask, val_mask, n_classes)
    print("classify_by_multi_prototypes Val Accuracy {:.4f}".format(acc))
    pred_labels, acc =classify_by_multi_prototypes(logits, labels, train_mask, test_mask, n_classes)
    print("Test Accuracy {:.4f}".format(acc))

    pred_labels, acc = batchwise_greedy_transductive_classification(logits, labels, train_mask, val_mask, n_classes)
    print("batchwise_greedy_transductive_classification Val Accuracy {:.4f}".format(acc))
    pred_labels, acc = batchwise_greedy_transductive_classification(logits, labels, train_mask, test_mask, n_classes)
    print("Test Accuracy {:.4f}".format(acc))

    cls = []
    cls.append(classify_by_similarity)
    cls.append(classify_by_nearest_neighbor )
    cls.append(classify_by_weighted_prototype )
    cls.append(classify_by_multi_prototypes)



    #stack_ensemble(cls, logits, labels, train_mask, val_mask, test_mask, n_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--seed", type=int, default=42,
            help="seed")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--alpha", type=float, default=0.5,
            help="balance between local and neighbors")
    parser.add_argument("--split", type=int, default=0,
            help="which split")
    parser.add_argument("--use-feature", type=str, default='node_label',
            help="which split")
    args = parser.parse_args()
    print(args)

    seed_value = args.seed 
    torch.manual_seed(seed_value)
    main(args)
