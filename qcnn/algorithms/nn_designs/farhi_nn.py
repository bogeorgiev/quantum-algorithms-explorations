import cirq
import numpy as np
from matplotlib import pyplot as plt
import torch
import math
import random
from cirq import Simulator

simulator = Simulator()

def choose_subset(n):
    a = np.random.randint(0, high=2, size=n)
    if np.sum(a) % 2 == 0:
        a[0] = 1 - a[0]
    subset = []
    for i in range(n):
        if a[i] == 1:
            subset += [i]
    return a, subset


def subset_parity(subset, input_b, n):
    s = sum(input_b[subset])
    if s % 2 == 0:
        l_z = 1.0
    else:
        l_z = -1.0
    return l_z

def prepare_initial_state(b, qubits, n):
    for i in range(n):
        if b[i] == 1:
            yield cirq.X(qubits[i])

def measure_Y(qubit):
    yield cirq.ZPowGate(exponent=1.5)(qubit), cirq.H(qubit)

def mod2pi(theta):
    new_theta = np.tile(0.0, len(theta))
    for i in range(len(theta)):
        a = theta[i] % (2 * np.pi)
        if np.pi < a < 2*np.pi:
            a = a - 2 * np.pi
        new_theta[i] = a
    return new_theta

def compute_est(hist, num_reps, kkey):
    return (hist(key=kkey)[0] - hist(key = kkey)[1]) / num_reps


n = 4
length = n+1
num_reps = 200

a, subset = choose_subset(n)
circuit = cirq.Circuit()

b = np.random.randint(0, high=2, size=n)
l_z = subset_parity(subset, b, n)

print(a, b)

qubits = [cirq.GridQubit(i, 0) for i in range(length-1)]
readout = cirq.GridQubit(length-1, 0)
circuit.append([prepare_initial_state(b, qubits, n)])
circuit.append([cirq.Rx(-np.pi/2)(readout)])

for i in range(n):
    cX = cirq.ControlledGate(cirq.Rx(-np.pi * a[i]))
    circuit.append([cX(qubits[i], readout)])

circuit.append([measure_Y(readout), cirq.measure(readout, key='demo')])
print(circuit)

results = simulator.run(circuit, repetitions=num_reps)
estimate = compute_est(results.histogram, num_reps, 'demo')

print("computed Value of function", estimate)
