from grover_superposition import superposition
import cirq
import numpy as np
import random
import torch.nn as nn
import torch


class QCNN:
    """
    # contains qubit-register for holding the current state of information (holdqubits), on which there is an
    # operation according to the measurement of the batchqubits
    # another register is manipulated to represent the current batch
    # every batch has its own parameters
    # batch dimension must be = 2^(number of batchqubits)
    # required modules: cirq, numpy, random
    # class has to import meethod superposition from grover_superposition
    # the convolution itself creates the mini-circuit, the class does not hold
    # the qubits which are responsible ffor the convolution
    """

    def __init__(self, shift_size, examples_per_component):
        # self.examples = examples
        self.type = "QCNN"
        self.num_window_qubits = 4
        self.num_hold_qubits = 4
        self.num_windows = 169
        self.parameters = np.array([
            [
                [np.random.rand() * 2 * np.pi for j in range(self.num_window_qubits)],
                [np.random.rand() * 2 * np.pi for k in range(self.num_hold_qubits)],
                [np.random.rand() * 2 * np.pi for l in range(self.num_hold_qubits)]
            ]
            for i in range(self.num_windows)
        ])

        # hyperparameters for gradient descent methods:
        # size of shift to approximate gradient
        self.shift_size = shift_size
        # how many examples are evaluated for each component of the gradient
        self.examples_per_component = examples_per_component

    # needs to be fed the holdqubits, to conduct the final measurement
    def convolution(self, index, window):
        # initiation
        qubits = cirq.LineQubit.range(self.num_window_qubits)
        circuit = cirq.Circuit()
        simulator = cirq.Simulator()

        # encode the current window (vector) into a superposition state
        circuit.append(superposition(window, qubits))

        # first layer w/o measurement
        circuit.append(cirq.Ry(self.parameters[index][0][i]).on(qubits[i]) for i in range(self.num_window_qubits))
        circuit.append(cirq.CZ(qubits[2 * i], qubits[2 * i + 1]) for i in range(int(self.num_window_qubits / 2)))
        circuit.append(cirq.CZ(qubits[1], qubits[2]))

        # Bell measurement, removing every second qubits from contention
        # current: measure 0,2 then conditional Z gate on 1,3
        circuit.append(cirq.measure(qubits[0], qubits[2]))
        circuit.append([cirq.CNOT(qubits[0], qubits[1]), cirq.CNOT(qubits[2], qubits[3])])

        # fully connected layer
        circuit.append(cirq.CZ(qubits[1], qubits[3]))

        # measure final qubits
        circuit.append(cirq.measure(qubits[1], qubits[3]))

        # simulate the partial circuit and return the measured values
        result = simulator.run(circuit)
        # return measurement on qubit 1, measurement on qubit 3
        return result.measurements['1,3'][0][0], result.measurements['1,3'][0][1]

    # method going through one training example
    def feedforward(self, data, repetitions=1):
        """
        data: #windows X window_size
        """
        holdqubits = cirq.LineQubit.range(self.num_hold_qubits)
        maincircuit = cirq.Circuit()
        mainsim = cirq.Simulator()

        # prepare holdqubits by putting them into a superposition
        for i in range(self.num_hold_qubits):
            maincircuit.append(cirq.H(holdqubits[i]))

        # enumerate over windows:
        for index, window in enumerate(data):
            measurement1, measurement2 = self.convolution(index, window)

            # measurement = True refers to state |1>, False to |0>
            # just make them CNOT for now
            # bit of redundancy here are double-true negates itself again
            if measurement1:
                for i in range(self.num_hold_qubits):
                    maincircuit.append(cirq.Ry(self.parameters[index][1][i])(holdqubits[i]))
            if measurement2:
                for i in range(self.num_hold_qubits):
                    maincircuit.append(cirq.Ry(self.parameters[index][2][i])(holdqubits[i]))
                    maincircuit.append(cirq.X(holdqubits[i]))

        # after all windows are passed through, we measure
        maincircuit.append(cirq.measure(*holdqubits, key='x'))
        result = mainsim.run(maincircuit, repetitions=repetitions)
        # print(result.histogram(key='x')[1])
        return result.histogram(key='x')  # result.measurements['0,1,2,3'][0]

    def prepare_histogram(self, histogram):
        result = []
        s = 0
        for i in range(10):
            s += histogram[i]
        if s == 0:
            s = 1
        for i in range(10):
            result += [float(histogram[i]) / float(s)]
        return result

    # runs the circuit for a set of examples
    def evaluate_error(self, sample, classification, repetitions):

        result = self.feedforward(sample, repetitions=repetitions)
        result = self.prepare_histogram(result)
        result = torch.tensor(result).unsqueeze(0)
        classification = classification.unsqueeze(0)
        loss_func = nn.CrossEntropyLoss()

        return loss_func(result, classification)

    def evaluate_result(self, result):
        number = 0
        for i in range(len(result)):
            if result[i]:
                number += 2 ** i
        return number

