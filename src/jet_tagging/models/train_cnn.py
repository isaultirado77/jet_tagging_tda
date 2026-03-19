import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    for x, y in tqdm(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
