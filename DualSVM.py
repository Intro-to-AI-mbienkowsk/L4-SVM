import numpy as np
from util import *
from sklearn import datasets


class DualSVM:
    def __init__(self, c=DEFAULT_C,
                 kernel=KERNEL.GAUSSIAN,
                 sigma=DEFAULT_SIGMA,
                 degree=DEFAULT_DEGREE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 epochs=DEFAULT_EPOCHS,
                 num_to_recognize=DEFAULT_NUM_TO_RECOGNIZE):
        ...

    def _gaussian_kernel(self, x1, x2):
        ...

    def _polynomial_kernel(self, x1, x2, degree):
        return (x1 @ x2 + POLYNOMIAL_KERNEL_CONSTANT_TERM) ** degree

    def fit(self, x, y):
        ...

    def predict(self, x):
        ...

    def score(self, x, y):
        ...
