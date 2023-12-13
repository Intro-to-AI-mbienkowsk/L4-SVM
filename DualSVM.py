import numpy as np
from util import *
from scipy.spatial.distance import cdist
from sklearn import datasets


class DualSVM:
    def __init__(self,
                 training_data,
                 training_labels,
                 c=DEFAULT_C,
                 kernel=KERNEL.GAUSSIAN,
                 sigma=DEFAULT_SIGMA,
                 degree=DEFAULT_DEGREE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 epochs=DEFAULT_EPOCHS,
                 num_to_recognize=DEFAULT_NUM_TO_RECOGNIZE):

        self.sigma = sigma
        self.c = c
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num = num_to_recognize

        self.training_x = training_data
        self.training_y = self.process_labels(training_labels)
        self.dimension = np.size(training_labels)

        self.kernel_fun = kernel
        self.degree = degree

        ## todo
        self.alpha = np.random.random(self.dimension)
        self.bias = 0

    def process_labels(self, y):
        """Transforms the decimal labels into binary ones (it either looks for the number or does not)"""
        return np.where(y == self.num, 1, -1)

    def kernel(self, x1, x2):
        return self._gaussian_kernel(x1, x2) \
            if self.kernel_fun == KERNEL.GAUSSIAN \
            else self._polynomial_kernel(x1, x2)

    def _gaussian_kernel(self, x1, x2):
        return np.exp(-cdist(x1, x2) / 2 * self.sigma ** 2)

    def _polynomial_kernel(self, x1, x2):
        return (x1 @ x2 + POLYNOMIAL_KERNEL_CONSTANT_TERM) ** self.degree

    def fit(self):
        x, y = self.training_x, self.training_y
        y_pair_kernel_product = np.outer(y, y) * self.kernel(x, x)
        for i in range(self.epochs):
            gradient = np.ones(self.dimension) - y_pair_kernel_product @ self.alpha
            self.alpha = self.alpha + self.learning_rate * gradient
            self.alpha[self.alpha > self.c] = self.c
            self.alpha[self.alpha < 0] = 0

        s_vect_indices = np.where(self.c > self.alpha > 0)[0]
        self.bias = np.mean(y[s_vect_indices] - np.dot((self.alpha * y), self.kernel(x, x[s_vect_indices])))

    def predict(self, x):
        return np.dot((self.alpha * self.training_y), self.kernel(self.training_x, x)) + self.bias

    def score(self, x, y):
        return np.mean(y == self.predict(x))

    def hinge_loss(self):
        ...
