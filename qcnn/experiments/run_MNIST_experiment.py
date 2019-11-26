import sys
sys.path.append("/home/bogdan/Projects/Quantum Computing/quantum-algorithms/quantum-algorithms/")
import argparse
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import cirq
from cirq import Simulator
import math
import numpy as np

## CUSTOM METHODS

from grover_superposition import superposition
from mnist_classifier import prepare_windows, evaluate_gradient
from qmlclasses import QCNN


## EXPERIMENT PARAMETERS
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epoch_batches', type=int, default=50, metavar='N',
                    help='number of batches per epoch (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 3000)')
parser.add_argument('--lr', type=float, default=5.0e-5, metavar='LR',
                    help='learning rate (default: 5.0e-5)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--window-size-x', type=int, default=4,
                    help='window x-dim')
parser.add_argument('--window-size-y', type=int, default=4,
                    help='window y-dim')
parser.add_argument('--window-stride', type=int, default=2,
                    help='window stride')
parser.add_argument('--gradient-shift', type=float, default=1.0e-1,
                    help='parameter shift')
parser.add_argument('--num-grad-iter', type=int, default=40,
                    help='examples per gradient component')
parser.add_argument('--examples-per-comp', type=int, default=40,
                    help='examples per gradient component')
parser.add_argument('--reps', type=int, default=20,
                    help='repetitions')


args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


# FRAMEWORK TO RUN THE EXPERIMENT

# initiate QCNN
qcnn = QCNN(args.gradient_shift, args.examples_per_comp)

loss_save = np.zeros(args.epochs)

for i in range(args.epochs):

    for j in range(args.epoch_batches):

        for batch_idx, data in enumerate(train_loader):
            batch, y = data[0], data[1]
            windows = prepare_windows(batch, args.batch_size, args.window_size_x, args.window_size_y,
                    args.window_stride)
            qcnn.parameters -= args.lr * evaluate_gradient(qcnn, windows, y, args.reps)

    # test current parameters
    epoch_error = 0.
    for batch_idx, data in enumerate(test_loader):
        batch, y = data[0], data[1]
        windows = prepare_windows(batch, args.batch_size, args.window_size_x, args.window_size_y,
                              args.window_stride)
        epoch_error += qcnn.evaluate_error(batch, y)

    print('In epoch {} the error is: {}'.format(i, epoch_error))
    loss_save[i] = epoch_error

print(loss_save)
