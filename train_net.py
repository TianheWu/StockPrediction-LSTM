import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np

from utils.data.stock_data import StockDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train LSTM')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    return args


def train():
    pass


def run():
    args = parse_args()
    dataset = StockDataset(args.dataset_path)


if __name__ == '__main__':
    run()
