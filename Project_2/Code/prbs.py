import numpy as np
import matplotlib.pyplot as plt


def prbs31(code):
    for i in range(32):
        next_bit = ~((code>>30) ^ (code>>27))&0x01
        code = ((code<<1) | next_bit) & 0xFFFFFFFF
    return code


def generate_prbs31(size, graph=False):
    seed = 1 # Initial value of seed
    sequence = []

    for _ in range(size):
        seed = prbs31(seed)
        sequence.append(seed & 1)

    if graph:
        plt.plot(sequence)
        plt.title('PRBS31')
        plt.xlabel('Samples')
        plt.ylabel('Bit')
        plt.show()

    return sequence


def generate_prbs_input(size, min_value, max_value, graph=False):
    # Generate PRBS
    prbs = generate_prbs31(size)

    # Scale PRBS
    scaled_prbs_signal = min_value + max_value * prbs

    if graph:
        # Graph Signal
        plt.plot(scaled_prbs_signal)
        plt.title('PRBS')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.show()

    return scaled_prbs_signal