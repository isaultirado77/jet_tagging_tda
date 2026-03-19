import argparse

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    DataLoader, 
    random_split
)

from jet_tagging.data.datasets import (
    JetImageDataset,
    PersistenceImagesDataset, 
    PersistenceImagesStackedDataset
)

from jet_tagging.models.cnn import (
    JetCNN, 
    PICNN, 
    PICNN2C
)

from jet_tagging.models.train_cnn import train_epoch

from jet_tagging.models.evaluate import (
    evaluate,
    evaluate_auc,
    compute_roc
)

from jet_tagging.config import (
    RESULTS_DIR, 
    MERGED_DATASETS_DIR
)

from jet_tagging.utils import (
    create_run_dir, 
    save_json
)

from jet_tagging.plotting import (
    plot_training, 
    plot_roc
)


def get_dataset(mode, path, n_sample, pi_mode=None):

    if mode == 'jet':
        return JetImageDataset(path, n_sample=n_sample)

    elif mode == 'pi':

        if pi_mode == 'H0':
            return PersistenceImagesDataset(path, pi_key='pi_H0', n_sample=n_sample)

        elif pi_mode == 'H1':
            return PersistenceImagesDataset(path, pi_key='pi_H1', n_sample=n_sample)

        elif pi_mode == 'H0H1':
            return PersistenceImagesStackedDataset(path, n_sample=n_sample)

        else:
            raise ValueError("Invalid pi_mode")

    else:
        raise ValueError("mode must be 'jet' or 'pi'")


def get_model(mode, device, pi_mode=None):

    if mode == 'jet':
        return JetCNN().to(device)

    elif mode == 'pi':

        if pi_mode in ['H0', 'H1']:
            return PICNN().to(device)

        elif pi_mode == 'H0H1':
            return PICNN2C().to(device)

        else:
            raise ValueError("Invalid pi_mode")

    else:
        raise ValueError


def parse_args(): 
    parser = argparse.ArgumentParser(
        description='Train Jet Tagging CNN model'
    )

    parser.add_argument(
        '--mode', 
        type=str, 
        default='jet', 
    )

    parser.add_argument(
        '--pi_mode', 
        type=str, 
        default='H0', 
        choices=['H0', 'H1', 'H0H1'],
        required=False
    )

    parser.add_argument(
        '--data_path', 
        type=str, 
        default=str(MERGED_DATASETS_DIR / "jet_images_merged_100000.h5")
    )

    parser.add_argument(
        '--n_sample', 
        type=int, 
        default=5000
    )

    parser.add_argument(
        '--batch_size', 
        type=int,
        default=64
    )

    parser.add_argument(
        '--epochs',
        type=int, 
        default=5
    )

    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-3
    )

    return parser.parse_args()


def main():

    # Config
    args = parse_args() 
    mode = args.mode  # 'jet', 'pi'
    pi_mode = args.pi_mode  if args.mode == 'pi' else None  # 'H0', 'H1', 'H0H1'
    data_path = args.data_path
    n_sample = args.n_sample

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Device: {device}')
    print(f'Mode: {mode}')
    if pi_mode: print(f'Pi-Mode: {pi_mode}')

    # Dataset
    dataset = get_dataset(mode, data_path, n_sample, pi_mode)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = get_model(mode, device, pi_mode)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accuracies = []
    val_aucs = []

    # Train loop
    for epoch in range(epochs):

        loss = train_epoch(model, train_loader, optimizer, criterion, device)        
        acc = evaluate(model, val_loader, device)
        auc = evaluate_auc(model, val_loader, device)

        train_losses.append(loss)
        val_accuracies.append(acc)
        val_aucs.append(auc)

        print(f'[Epoch {epoch}] loss={loss:.4f} acc={acc:.4f} auc={auc:.4f}')

    # Results
    run_dir = create_run_dir(mode)

    config = vars(args)    

    metrics = {
        'train_loss': train_losses, 
        'val_accuracy': val_accuracies, 
        'val_auc': val_aucs
    }
    roc_data = compute_roc(model, val_loader, device)

    plot_roc(roc_data['fpr'], roc_data['tpr'], run_dir / "roc.png")
    plot_training(metrics, run_dir / "training.png")
    
    np.savez(run_dir / "rpc_data.npz", **roc_data)
    save_json(metrics,run_dir / "metrics.json")
    save_json(config, run_dir / "config.json")

if __name__ == "__main__":
    main()
