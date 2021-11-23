import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils.data.stock_data import StockDataset
from model.lstm import Lstm
from torch.utils.data import DataLoader


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM')

    # Dataset
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Fraction of data for training (remainder is validation)')

    # Output
    parser.add_argument('--outdir', type=str, default='output/eval/', help='Training Output Directory')
    parser.add_argument('--description', type=str, default='', help='Training description')

    # Evaluation
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataset workers')
    parser.add_argument('--weight-path', type=str, help='LSTM model weights')

    args = parser.parse_args()
    return args


def validate(val_data, train_data, net):
    axis_x1 = np.linspace(1, 1180, 1180)
    res_label = []
    res_pred = []
    for x, y in train_data:
        res_label.extend(list(y.detach().numpy()))
        res_pred.extend(list(net(x.unsqueeze(0)).squeeze(0).detach().numpy().squeeze()))

    for x, y in val_data:
        res_label.extend(list(y.detach().numpy()))
        res_pred.extend(list(net(x.unsqueeze(0)).squeeze(0).detach().numpy()[0]))
    
    print(len(res_pred))
    print(len(res_label))
    
    plt.plot(axis_x1, res_label, label='label')
    plt.plot(axis_x1, res_pred, label='pred')
    plt.legend()
    plt.show()


def run():
    args = parse_args()
    
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
    # device = torch.device("cuda:0")
    input_size = 8
    net = Lstm(input_size)
    net = net.double()
    net.load_state_dict(torch.load(args.weight_path, map_location=torch.device('cpu')), strict=False)
    net.eval()
    print("Done")
    validate(val_data, train_data, net)


if __name__ == '__main__':
    run()
