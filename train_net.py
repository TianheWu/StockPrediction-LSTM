import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from utils.data.stock_data import StockDataset
from model.lstm import Lstm
from torch.utils.data import DataLoader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM')

    # Dataset
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Fraction of data for training (remainder is validation)')

    # Output
    parser.add_argument('--outdir', type=str, default='output/', help='Training Output Directory')
    parser.add_argument('--description', type=str, default='', help='Training description')

    # Training
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')

    args = parser.parse_args()
    return args


def validate(val_data, net, criterion, device, epoch):
    net.eval()
    loss_sum = 0
    batch_idx = 0
    for x, y in val_data:
        batch_idx += 1
        x = x.to(device)
        y = y.to(device)
        pred = net(x.unsqueeze(0))
        loss = criterion(pred, y.unsqueeze(0))
        loss_sum += loss.item()
    
    ret = loss_sum / batch_idx
    print('Eval Epoch: {}, Loss: {:0.4f}'.format(epoch, ret))

    return ret


def train(epoch, train_data, net, optimizer, criterion, device):
    net.train()
    loss_sum = 0
    batch_idx = 0
    for x, y in train_data:
        batch_idx += 1
        x = x.to(device)
        y = y.to(device)
        pred = net(x.unsqueeze(0))
        loss = criterion(pred, y.unsqueeze(0))
        loss_sum += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ret = loss_sum / batch_idx
    print('Epoch: {}, Loss: {:0.4f}'.format(epoch, ret))

    return ret


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
    # x, y = train_dataset[0]
    # print(x.shape)
    # print(x)

    val_data = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    print("Done")
    print("Loading Network...")
    device = torch.device("cuda:0")
    input_size = 8
    net = Lstm(input_size)
    net = net.to(device).double()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    print("Done")

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_results = train(epoch, train_data, net, optimizer, criterion, device)
        test_results = validate(val_data, net, criterion, device, epoch)
        if test_results < best_val_loss:
            best_val_loss = test_results
            torch.save(net, os.path.join(save_folder, 'epoch_%03d_loss_%0.4f' % (epoch, best_val_loss)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%03d_loss_%0.4f_statedict.pt' % (epoch, best_val_loss)))


if __name__ == '__main__':
    run()
