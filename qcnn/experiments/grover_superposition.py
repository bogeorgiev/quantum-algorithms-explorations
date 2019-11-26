import cirq
import numpy as np

from random import randint


# goal is putting a classical vector into a quantum mechanical superposition
# the procedure follows 'Grover and Rudolph (2008)'
# length of vector must be smaller/equal 2^(number of qubits)
# the function automatically lengthens x with an array of zeros to the next power of 2
def superposition(x, qubits):
    N = len(qubits)
    x = np.append(x, np.zeros(2 ** N - len(x)))                     # adjust length of x
    powers = np.array([2 ** (N - 1 - i) for i in range(N)])
    for level in range(N):           # go through levels (by definition -> level refers to the currently rotated qubit)
        level += 1
        if level == 1:  # special circuit for level 1 --> no controlled operation is required
            angle = determine_angle(x, 0, len(x) - 1)
            yield cirq.Ry(2 * angle)(qubits[0])
        else:
            permutations = give_binary_vectors(level, N)         # each perm of length (level-1) represents a branch
            for current, perm in enumerate(permutations):        # iterate over branches
                xl = np.dot(perm, powers)                        # determine boundaries of integral-function
                if current == len(permutations) - 1:
                    xr = 2 ** N - 1
                else:
                    xr = np.dot(permutations[current + 1], powers) - 1
                angle = determine_angle(x, int(xl), int(xr))     # call integral-function to determine rotation angle

                for k in range(level - 1):                       # prepare desired state |xx...x0> to |11...10>
                    if perm[k] == 0.:
                        yield cirq.X(qubits[k])

                controlled_by_qubits = [qubits[i] for i in range(level-1)]

                # controlled rotation

                yield cirq.Ry(2 * angle).controlled_by(*controlled_by_qubits).on(qubits[level-1])

                for k in range(level - 1):                       # 'unprepare' state
                    if perm[k] == 0.:
                        yield cirq.X(qubits[k])


# xl and xr are first and last index included (!) in the range within x of interest
def determine_angle(x, xl, xr):
    cut = np.int(xl + (xr - xl)/2)
    # prevent error if all current elements are zero
    if np.sum(x[xl:xr+1]) == 0:
        y = 0.5
    else:
        y = np.sum(x[xl:cut+1]) / np.sum(x[xl:xr+1])
    return np.arccos(np.sqrt(y))


# gives all binary vectors on length (level - 1)
def give_binary_vectors(level, N):
    final = []
    for k in range(2**(level-1)):           # go through all numbers (0 to 2^^(m-1))
        number = 2**(level-1) - k - 1
        x = np.zeros(N)
        for j in range(level-1):            # compute binary of number
            x[j], number = modulo_two_addition(number, 2**(level-j-2))
        final.append(np.flip(x))
    return np.flip(final)


# just for convenience
def modulo_two_addition(x, y):
    return (x-x % y)/y, x % y
