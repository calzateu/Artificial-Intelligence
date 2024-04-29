import numpy as np


def calculate_output_error_cross_entropy(targets, forwarded_inputs):
    # Binary cross entropy loss
    return -(targets * np.log(forwarded_inputs) + (1 - targets) * np.log(1 - forwarded_inputs))


def calculate_output_error_mse(targets, forwarded_inputs):
    # Mean squared error
    return 1/2*(forwarded_inputs - targets)**2


def derivate_output_error_cross_entropy(targets, forwarded_inputs):
    # If dimensions don't match then throw an error
    if targets.shape != forwarded_inputs.shape:
        raise ValueError(f"The dimensions of the targets and the output of the layer "
                         f"don't match {targets.shape} != {forwarded_inputs.shape}.")
    # Derivative of binary cross entropy loss
    return - targets / forwarded_inputs + (1 - targets) / (1 - forwarded_inputs)


def derivate_output_error_mse(targets, forwarded_inputs):
    # If dimensions don't match then throw an error
    if targets.shape != forwarded_inputs.shape:
        raise ValueError(f"The dimensions of the targets and the output of the layer"
                         f"don't match {targets.shape} != {forwarded_inputs.shape}.")
    # Derivative of mean squared error
    return forwarded_inputs - targets


def activation(z):
    return 1 / (1 + np.exp(-z))


def derivative_activation(z):
    return z * (1 - z)
