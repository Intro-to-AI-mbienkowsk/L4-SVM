import numpy as np
from sklearn import datasets


class DualSVM:
    def __init__(self, c, kernel, sigma, degree, learning_rate, epochs):
        ...

    def _rbf_kernel(self, x1, x2):
        ...

    def _polynomial_kernel(self, x1, x2, degree):
        ...

    def fit(self, x, y):
        ...

    def predict(self, x):
        ...

    def score(self, x, y):
        ...

