import numpy as np
import cirq
import random
import sympy
from matplotlib import pyplot as plt

length = 3

qubits = [cirq.GridQubit(i, j) for i in range(length) for j in range(length)]

#circuit = cirq.Circuit()
#circuit.append([cirq.H(q) for q in qubits if (q.row + q.col) % 2 == 0], strategy=cirq.InsertStrategy.EARLIEST)
#circuit.append([cirq.X(q) for q in qubits if (q.row + q.col) % 2 == 1], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

def rot_x_layer(length, half_turns):
    rot = cirq.XPowGate(exponent=half_turns)
    for i in range(length):
        for j in range(length):
            yield rot(cirq.GridQubit(i, j))

def rot_z_layer(h, half_turns):
    gate = cirq.ZPowGate(exponent=half_turns)
    for i, h_row in enumerate(h):
        for j, h_ij in enumerate(h_row):
            if h_ij == 1:
                yield gate(cirq.GridQubit(i, j))

def rot_11_layer(jr, jc, half_turns):
    gate = cirq.CZPowGate(exponent=half_turns)
    for i, jr_row in enumerate(jr):
        for j, jr_ij in enumerate(jr_row):
            if jr_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i + 1, j))
            yield gate(cirq.GridQubit(i, j),
                    cirq.GridQubit(i + 1, j))
            if jr_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i + 1, j))

    for i, jc_row in enumerate(jc):
        for j, jc_ij in enumerate(jc_row):
            if jc_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i, j + 1))
            yield gate(cirq.GridQubit(i, j),
                    cirq.GridQubit(i, j + 1))
            if jc_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i, j + 1))


def one_step(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    length = len(h)
    yield rot_x_layer(length, x_half_turns)
    yield rot_z_layer(h, h_half_turns)
    yield rot_11_layer(jr, jc, j_half_turns)


def energy_func(length, h, jr, jc):
    def energy(measurements):
        meas_list_of_lists = [measurements[i * length:(i+1) * length] for i in range(length)]
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)
        tot_energy = np.sum(pm_meas * h)
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i+1, j]

        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j+1]
        return tot_energy
    return energy

def obj_func(result):
    energy_hist = result.histogram(key='x', fold_func=energy_func(3, h, jr, jc))
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions

def rand2d(rows, cols):
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]

def random_instance(length):
    h = rand2d(length, length)
    jr = rand2d(length - 1, length)
    jc = rand2d(length, length - 1)
    return (h, jr, jc)

h, jr, jc = random_instance(3)


alpha = sympy.Symbol("alpha")
beta = sympy.Symbol("beta")
gamma = sympy.Symbol("gamma")

circuit = cirq.Circuit()
circuit.append(one_step(h, jr, jc, alpha, beta, gamma), strategy=cirq.InsertStrategy.EARLIEST)
circuit.append(cirq.measure(*qubits, key='x'))
print(circuit)

resolver = cirq.ParamResolver({"alpha": 0.1, "beta": 0.3, "gamma": 0.7})
resolved_circuit = cirq.resolve_parameters(circuit, resolver)

sweep = (cirq.Linspace(key="alpha", start=0.1, stop=0.9, length=5)
        *cirq.Linspace(key="beta", start=0.1, stop=0.9, length=5)
        *cirq.Linspace(key="gamma", start=0.1, stop=0.9, length=5))

simulator = cirq.Simulator()
results = simulator.run(resolved_circuit, repetitions=1000)
print(results.histogram(key='x'))
print(results.histogram(key='x', fold_func=energy_func(3, h, jr, jc)))
print("Value of the objective function {}".format(obj_func(results)))

results = simulator.run_sweep(circuit, params=sweep, repetitions=100)
for result in results:
    print(result.params.param_dict, obj_func(result))

min = None
min_params = None

for result in results:
    value = obj_func(result)
    if min is None or value < min:
        min = value
        min_params = result.params

print("Minimum Objective: {}".format(min), "Optimal Parameters: {}".format(min_params))
