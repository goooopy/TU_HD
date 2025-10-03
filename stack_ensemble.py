import torch
import torch.nn as nn
import copy
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

class TwoLayerMLP(nn.Module):
    def __init__(self, n_classes):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # first layer: 4 -> 4
        self.fc2 = nn.Linear(128, n_classes)  # second layer: 4 -> 1

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU between layers
        x = self.fc2(x)          # final output (no activation here)
        return x
def stack_ensemble(cls,logits, labels, train_mask, val_mask, test_mask, n_classes):
    classify_by_similarity = cls[0]
    classify_by_nearest_neighbor = cls[1]
    classify_by_weighted_prototype = cls[2]
    classify_by_multi_prototypes = cls[3]

    pred_labels1,_ = classify_by_similarity(logits, labels, train_mask, val_mask,n_classes) 
    pred_labels2,_ = classify_by_nearest_neighbor(logits, labels, train_mask, val_mask,n_classes)
    pred_labels3,_ = classify_by_weighted_prototype(logits, labels, train_mask, val_mask,n_classes)
    pred_labels4,_ = classify_by_multi_prototypes(logits, labels, train_mask, val_mask,n_classes)

    #print(pred_labels1)
    #print(pred_labels2)
    #print(pred_labels3)
    #print(pred_labels4)
    pred_labels4 = pred_labels4.cuda()

    features_train = torch.stack([pred_labels1, pred_labels2, pred_labels3, pred_labels4], dim=1).float()
    print('features_train', features_train.size())
    labels_train = labels[val_mask]

    model = TwoLayerMLP(n_classes).cuda()
    dataset = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_acc = 0
    # ----- Training Loop -----
    for epoch in range(100):  # 10 epochs
        correct = 0
        total = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.cuda()
            batch_labels = batch_labels.cuda()
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
        train_acc = 100 * correct / total
        if train_acc > best_acc:
            best_acc = train_acc
            best_dict = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
        print('train_acc', train_acc)
    model.load_state_dict(best_dict)

    pred_labels1,_ = classify_by_similarity(logits, labels, train_mask, test_mask,n_classes) 
    pred_labels2,_ = classify_by_nearest_neighbor(logits, labels, train_mask, test_mask,n_classes)
    pred_labels3,_ = classify_by_weighted_prototype(logits, labels, train_mask, test_mask,n_classes)
    pred_labels4,_ = classify_by_multi_prototypes(logits, labels, train_mask, test_mask,n_classes)
    pred_labels4 = pred_labels4.cuda()

    features_test = torch.stack([pred_labels1, pred_labels2, pred_labels3, pred_labels4], dim=1).float()
    labels_test = labels[test_mask]

    total = 0
    with torch.no_grad():
        outputs = model(features_test)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels_test).sum().item()
        total = labels_test.size(0)
        test_acc = 100 * correct / total
        print('test_acc', test_acc)
