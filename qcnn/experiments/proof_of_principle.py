#import sys
#sys.path.append("/home/bogdan/Projects/Quantum Computing/quantum-algorithms/quantum-algorithms/")
import argparse
import torch
from torchvision import datasets, transforms
import torchvision
from numpy import linalg as LP

import cirq
# import math
import numpy as np
import torch.nn as nn

## CUSTOM METHODS

from grover_superposition import superposition
from data_loader import CroppedRescaledMNISTDataset
from torch.utils.data import DataLoader

# define network for proof of principle
class QCNN_POP:

    def __init__(self):
        self.type = "QCNN_POP"
        self.parameters = np.array([np.random.rand() * 2 * np.pi for i in range(6)])
        self.num_qubits = 4


    # data must be np.array
    def forward(self, data, repetitions):
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()
        simulator = cirq.Simulator()

        circuit.append(superposition(data, qubits))
        circuit.append([cirq.Ry(self.parameters[i])(qubits[i]) for i in range(self.num_qubits)])
        circuit.append(cirq.Ry(self.parameters[0]).controlled_by(qubits[0]).on(qubits[1]))
        circuit.append(cirq.Ry(self.parameters[1]).controlled_by(qubits[2]).on(qubits[3]))
        circuit.append(cirq.Ry(self.parameters[2]).controlled_by(qubits[1]).on(qubits[2]))
        circuit.append(cirq.Ry(self.parameters[3]).controlled_by(qubits[3]).on(qubits[0]))

        circuit.append(cirq.measure(qubits[0], qubits[2]))

        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.CNOT(qubits[2], qubits[3]))
        circuit.append(cirq.CZ(qubits[1], qubits[3]))
        circuit.append(cirq.Ry(self.parameters[4]).controlled_by(qubits[1]).on(qubits[3]))

        circuit.append(cirq.measure(qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[3]))
        circuit.append(cirq.Ry(self.parameters[5])(qubits[3]))
        circuit.append(cirq.measure(qubits[3]))

        result = simulator.run(circuit, repetitions=repetitions)

        return result.histogram(key ='3')[0], result.histogram(key='3')[1]

def get_cross_entropy_for_sample(qcnn_pop, example, classification, repetitions):
    distribution = np.array([qcnn_pop.forward(example, repetitions)][0])
    # normalize
    distribution = np.divide(distribution, repetitions)
    actual_result = np.zeros(2)
    actual_result[int(classification)] = 1.
    return cross_entropy(softmax(actual_result), softmax(distribution))
    #return nn.CrossEntropyLoss(distribution, actual_result)


def run_test(qcnn, batch_size, repetitions, batch, classification):
    missclassifications = 0.
    for i in range(batch_size):
        # loss += get_cross_entropy_for_sample(qcnn, batch[i], classification[i], repetitions)
        # determine missclassifications:
        distribution = np.array([qcnn.forward(batch[i], repetitions)][0])
        # print('test distribution: ', distribution)
        # print(distribution)
        missclassifications += repetitions - distribution[int(classification[i])]
    return missclassifications/(batch_size*repetitions)


def create_batch(pool, size):
    # PROBLEM SPECIFIC NUMBER 16
    batch = np.zeros((size, 16))
    #for i in range(size):
    #    batch[i,:] = pool[np.random.randint(2)]
    batch[0,:] = pool[0]
    batch[1,:] = pool[1]
    classification = batch[:, 3]
    return batch, classification


def get_learning_rate(init_rate, missclassification_percentage):
    rate = init_rate * (1. - np.exp(-missclassification_percentage))
    return rate


def softmax(x):
    return np.divide(np.exp(x), sum(np.exp(x)))


def cross_entropy(x,y):
    return -np.dot(x, np.log(y))


def determine_gradient(qcnn, batch_size, shift_size, repetitions_per_example, batch, classification):
    gradient = np.zeros(len(qcnn.parameters))
    for j in range(batch_size):
        gradient_component = 0.
        for i in range(len(qcnn.parameters)):
            qcnn.parameters[i] = qcnn.parameters[i] + shift_size
            gradient_component += get_cross_entropy_for_sample(
                qcnn, batch[j], classification[j], repetitions_per_example)
            #print('first: ' , gradient_component)
            qcnn.parameters[i] = qcnn.parameters[i] - 2*shift_size
            gradient_component -= get_cross_entropy_for_sample(
                qcnn, batch[j], classification[j], repetitions_per_example)
            #print('second: ', gradient_component)
            qcnn.parameters[i] = qcnn.parameters[i] + shift_size
            gradient[i] += gradient_component
        #print('so i wrote into the gradient: ', gradient[i])

    x = gradient/(2* shift_size * batch_size)
    return gradient/LP.norm(gradient)


# ARGUMENT PARSER
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-training-batches', type=int, default=5, metavar='N',
                    help='number of training batches per epoch (default: 100)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 3000)')
parser.add_argument('--lr', type=float, default=4., metavar='LR',
                    help='learning rate (default: 0.5)')
parser.add_argument('--gradient-shift', type=float, default=1.,
                    help='parameter shift')
parser.add_argument('--reps', type=int, default=30,
                    help='repetitions')
parser.add_argument('--batches-per-epoch', type=int, default=4, metavar='N',
                    help='defines the number of batches per epoch (default: 4')
parser.add_argument('--logging-interval', type=int, default=3)
parser.add_argument('--training-interval', type=int, default=3)
parser.add_argument('--test-repetitions', type=int, default=100)

args = parser.parse_args()

qcnn = QCNN_POP()

first_number = 3
second_number = 6

dataset = CroppedRescaledMNISTDataset([first_number, second_number], [1., 0.92])

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

loss_save = np.array([])

learning_rate = args.lr
gradient_shift = args.gradient_shift
gradient_shift = 0.001

lr = 0.5

for i in range(args.epochs):

    # training
    step = 0
    for batch_idx, data in enumerate(train_loader):
        data[0] = data[0].numpy()
        data[1] = data[1].numpy()
        print('batch_index: ', batch_idx)

        batch = data[0]
        y = data[1]

        for j in range(args.batch_size):
            if y[j] == first_number:
                y[j] = 0
            else:
                y[j] = 1

        gradient = determine_gradient(qcnn, args.batch_size, gradient_shift, args.reps, batch, y)
        # qcnn.parameters -= learning_rate * gradient
        qcnn.parameters -= lr * gradient
        if ((step + 1) % args.logging_interval == 0):
            print('parameter update: {}'.format(qcnn.parameters))

        if ((step + 1) % args.training_interval == 0):
            batch = np.zeros((args.batch_size, 16))
            y = np.zeros(args.batch_size)
            for batch_idx, data in enumerate(test_loader):
                if batch_idx > 0:
                    break
                batch = data[0]
                y = data[1]

                for j in range(args.batch_size):
                    if y[j] == first_number:
                        y[j] = 0
                    else:
                        y[j] = 1
                # testing
                missclassification_percentage = run_test(qcnn, args.test_batch_size, args.test_repetitions, batch, y)
                print('Batch index {}  Missclass-percentage {} percent.'.format(batch_idx, missclassification_percentage*100))
                loss_save = np.append(loss_save, missclassification_percentage)

            # update learning rate
            learning_rate = get_learning_rate(args.lr, missclassification_percentage)
            gradient_shift = get_learning_rate(args.gradient_shift, missclassification_percentage)

        step = step + 1


np.savetxt('test.out', loss_save, delimiter=',')   # X is an array










