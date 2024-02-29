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

def generate_prbs_signal(prbs_sequence, duration, sample_rate):
    time = np.arange(0, duration, 1/sample_rate)
    prbs_signal = np.tile(prbs_sequence, int(np.ceil(duration / len(prbs_sequence))))
    prbs_signal = prbs_signal[:len(time)]
    return prbs_signal

# Parámetros PRBS
order = 4
taps = [3, 0]

# Parámetros señal
duration = 1  # segundos
sample_rate = 1000  # Hz

# Generar PRBS
prbs_generator = PRBSGenerator(order, taps)
prbs_sequence = prbs_generator.generate_sequence(100)

# Generar señal persistently excitada
prbs_signal = generate_prbs_signal(prbs_sequence, duration, sample_rate)

# Visualizar la señal
plt.plot(prbs_signal)
plt.title('Señal Persistentemente Excitada (PRBS)')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

# Escala la señal entre 0 y 50
scaled_prbs_signal = 25 * prbs_signal + 25

# Visualiza la señal escalada
plt.plot(scaled_prbs_signal)
plt.title('Señal Escalada Persistentemente Excitada (PRBS)')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.show()

print(scaled_prbs_signal)

