import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve
)


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def evaluate_auc(model, loader, device):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.softmax(logits, dim=1)[:,1]

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_probs)

    return auc


def compute_roc(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1)[:, 1]

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'probs': all_probs,
        'labels': all_labels
    }
