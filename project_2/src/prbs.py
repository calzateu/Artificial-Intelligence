import matplotlib.pyplot as plt
import numpy as np
import random


def prbs31(code: int) -> int:
    """
    Generate a pseudo-random binary number based on the given src.

    Args:
        code (int): The initial src used to generate the PRBS.

    Returns:
        int: The pseudo-random binary number.
    """
    for i in range(32):
        next_bit = ~((code >> 30) ^ (code >> 27)) & 0x01
        code = ((code << 1) | next_bit) & 0xFFFFFFFF
    return code


def generate_prbs31(size: int, graph: bool = False) -> np.ndarray:
    """
    Generate a Pseudorandom Binary Sequence with a given size.

    Args:
        size (int): The size of the sequence to be generated.
        graph (bool, optional): Whether to plot the sequence as a graph. Defaults to False.

    Returns:
        sequence (np.ndarray): The generated Pseudorandom Binary Sequence.
    """
    seed = random.randint(0, size)  # Initial value of seed
    sequence = np.zeros(size)

    for i in range(size):
        seed = prbs31(seed)
        sequence[i] = seed & 1

    if graph:
        plt.plot(sequence)
        plt.title('PRBS31')
        plt.xlabel('Samples')
        plt.ylabel('Bit')
        plt.show()

    return sequence


def generate_prbs_input(size: int, min_value: int, max_value: int, graph: bool = False) -> np.ndarray:
    """
    Generate PRBS input signal based on the given size, min_value, and max_value.
    Optionally, display a graph of the signal if graph is set to True.

    Args:
        size (int): The size of the PRBS input signal.
        min_value (int): The minimum value of the PRBS input signal.
        max_value (int): The maximum value of the PRBS input signal.
        graph (bool, optional): Whether to display a graph of the PRBS input signal. Defaults to False.

    Returns:
        scaled_prbs_signal (np.ndarray): The generated PRBS input signal.
    """
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
