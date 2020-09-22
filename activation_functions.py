import matplotlib.pyplot as plt
import numpy as np

from constants import X_DIM


# np.maximum(a, b) returns maximum of a and b
def ReLU(x):
    return np.maximum(x, 0.)


def dReLU(x):
    return ReLU(x)


# nm.where(condition, if_true, else)
def LeakyReLU(x, k=0.2):
    return np.where(x >= 0, x, x * k)


def dLeakyReLU(x, k=0.2):
    return np.where(x >= 0, 1., k)

# Hiperbolic tangens
def Tanh(x):
    return np.tanh(x)


def dTanh(x):
    return 1. - Tanh(x)**2

# sigmoid function
def Sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dSigmoid(x):
    return Sigmoid(x) * (1. - Sigmoid(x))


def weight_initializer(in_channels, out_channels):
    scale = np.sqrt(2. / (in_channels + out_channels))
    return np.random.uniform(-scale, scale, (in_channels, out_channels))
