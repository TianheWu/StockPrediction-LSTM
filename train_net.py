import datetime
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from utils.data.stock_data import StockDataset
from model.lstm import Lstm
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM')

    # Dataset
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Fraction of data for training (remainder is validation)')

    # Output
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')
    parser.add_argument('--description', type=str, default='', help='Training description')

    # Training
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=50, help='Batches per Epoch')

    args = parser.parse_args()
    return args


def validate():
    pass


def train(epoch, train_data, net, optimizer, criterion, batches_per_epoch):
    batch_idx = 0
    result = 0
    while batch_idx < batches_per_epoch:
        for x, y in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            pred = net(x)
            loss = criterion(pred, y)
            if batch_idx % 10 == 0:
                print('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            result += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    result /= batch_idx

    return result


def run():
    args = parse_args()

    # Set-up output directories
    save_folder = args.outdir
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Loading Stock Dataset...")
    train_dataset = StockDataset(args.dataset_path, start_idx=0.0, end_idx=args.split)
    val_dataset = StockDataset(args.dataset_path, start_idx=args.split, end_idx=1.0)
    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    print("Done")
    print("Loading Network...")

    input_size = 8
    net = Lstm(input_size)
    net = net.double()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    print("Done")

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print("Beginning Epoch {:02d}".format(epoch))
        train_results = train(epoch, train_data, net, optimizer, criterion, args.batches_per_epoch)
        if train_results < best_val_loss or epoch == 0 or (epoch % 10) == 0:
            best_val_loss = train_results
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, train_results)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.2f_statedict.pt' % (epoch, train_results)))


if __name__ == '__main__':
    run()
