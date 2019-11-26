import sys

sys.path.append("/home/bogdan/Projects/Quantum Computing/quantum-algorithms/quantum-algorithms/")

import torch
import argparse
from torchvision import datasets, transforms
import torchvision

from qmlclasses import QCNN
import math
import numpy as np
from grover_superposition import superposition
from torch.utils.data import Dataset, DataLoader

import cirq
from cirq import Simulator


def prepare_windows(batch, batch_size, window_size_x, window_size_y, window_stride):
    pil_transformer = torchvision.transforms.ToPILImage(mode=None)
    tensor_transformer = torchvision.transforms.ToTensor()

    num_h_translates = math.floor((28 - window_size_x) / window_stride) + 1
    num_v_translates = math.floor((28 - window_size_y) / window_stride) + 1
    num_windows = num_h_translates * num_v_translates

    out = torch.zeros([batch_size, num_windows, window_size_x * window_size_y])

    for sample_idx in range(batch_size):
        sample = batch[sample_idx, 0, :, :]
        sample = pil_transformer(sample)
        for x in range(num_h_translates):
            for y in range(num_v_translates):
                current_window = torchvision.transforms.functional.crop(sample, x * window_size_x,
                                                                        y * window_size_y,
                                                                        window_size_x, window_size_y)
                current_window = tensor_transformer(current_window)
                current_window = current_window.view(-1)
                out[sample_idx, y * num_v_translates + x, :] = current_window
    return out.numpy()


def evaluate_gradient(qcnn, windows, y, repetitions):
    # initialize gradient
    gradient = np.zeros_like(qcnn.parameters)
    result = 0.
    batch = windows.shape[0]

    # goes through the matrices per window
    for i, window_parameters in enumerate(qcnn.parameters):
        for j, set_of_parameters in enumerate(window_parameters):
            for k, current_parameter in enumerate(set_of_parameters):
                # initialize gradient_component
                gradient_component = 0.
                # slightly shift this parameter
                qcnn.parameters[i][j][k] = (qcnn.parameters[i][j][k] + qcnn.shift_size) % 2.*np.pi
                for i in range(batch):
                    # determine quality of shifted parameter
                    gradient_component += qcnn.evaluate_error(windows[i, :, :], y[i],
                                                              repetitions=repetitions)
                # shift parameter to the other direction, then same procedure
                qcnn.parameters[i][j][k] = (qcnn.parameters[i][j][k] - 2 * qcnn.shift_size) % 2.*np.pi
                # second evaluation
                for i in range(batch):
                    gradient_component -= qcnn.evaluate_error(windows[i, :, :], y[i],
                                                              repetitions=repetitions)
                # return parameter to original value
                qcnn.parameters[i][j][k] = (qcnn.parameters[i][j][k] + qcnn.shift_size) % 2.*np.pi
                # normalize
                gradient_component /= 2 * qcnn.shift_size
                # put result into gradient
                gradient[i][j][k] = gradient_component

                # print('number of parameters trained: {}'.format(counter))
                # print('gradient component: ', gradient_component)

    return gradient


"""
windows = prepare_windows(batch)
#print(windows[15,:,:][0])
qcnn = QCNN(4, 4, 1.0e-1, 10, windows.shape[1], 0.1)
print('batch shape: ', batch.shape)
print('windows shape: ', windows.shape)
print('y shape: ', y.shape)
print(y)
print(windows)
#qcnn.parameters -= qcnn.learning_rate * evaluate_gradient(qcnn, batch, y)
results = evaluate_gradient(qcnn, batch, y)
print(results)
"""

