import numpy as np
import matplotlib.pyplot as plt

class PRBSGenerator:
    def __init__(self, order, taps):
        self.order = order
        self.taps = taps
        self.state = 1

    def shift(self):
        feedback = sum((self.state >> tap) & 1 for tap in self.taps) % 2
        self.state = ((self.state << 1) | feedback) & ((1 << self.order) - 1)
        return feedback

    def generate_sequence(self, length):
        sequence = []
        for _ in range(length):
            bit = self.shift()
            sequence.append(bit)
        return sequence

def generate_prbs_signal(size, min_value, max_value, order, taps, duration,
                         sample_rate, graph=False):
    # Generate PRBS
    prbs_generator = PRBSGenerator(order, taps)
    prbs_sequence = prbs_generator.generate_sequence(size)

    # Generate Signal
    time = np.arange(0, duration, 1/sample_rate)
    prbs_signal = np.tile(prbs_sequence, int(np.ceil(duration / len(prbs_sequence))))
    prbs_signal = prbs_signal[:len(time)]

    # Scale Signal
    #scaled_prbs_signal = 25 * prbs_signal + 25
    scaled_prbs_signal = min_value + max_value * prbs_signal

    if graph:
        # Graph Signal
        plt.plot(scaled_prbs_signal)
        plt.title('PRBS')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.show()

    return scaled_prbs_signal

