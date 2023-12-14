import numpy as np
from src.util import *
from scipy.spatial.distance import cdist


class DualSVM:
    def __init__(self,
                 training_data,
                 training_labels,
                 c=DEFAULT_C,
                 kernel=KERNEL.POLYNOMIAL,
                 sigma=DEFAULT_SIGMA,
                 degree=DEFAULT_DEGREE,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 epochs=DEFAULT_EPOCHS):
        self.sigma = sigma
        self.c = c
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_x = training_data
        self.training_y = training_labels
        self.dimension = np.size(training_labels)

        self.kernel_fun = kernel
        self.degree = degree

        ## todo
        self.alpha = np.random.random(self.dimension)
        self.bias = 0

    def kernel(self, x1, x2):
        return self._gaussian_kernel(x1, x2) \
            if self.kernel_fun == KERNEL.GAUSSIAN \
            else self._polynomial_kernel(x1, x2)

    def _gaussian_kernel(self, x1, x2):
        return np.exp(-cdist(x1, x2) / 2 * self.sigma ** 2)

    def _polynomial_kernel(self, x1, x2):
        return (x1.dot(x2.T) + POLYNOMIAL_KERNEL_CONSTANT_TERM) ** self.degree

    def fit(self):
        x, y = self.training_x, self.training_y
        y_pair_kernel_product = np.outer(y, y) * self.kernel(x, x)
        for i in range(self.epochs):
            gradient = np.ones(self.dimension) - y_pair_kernel_product @ self.alpha
            self.alpha = self.alpha + self.learning_rate * gradient
            self.alpha[self.alpha > self.c] = self.c
            self.alpha[self.alpha < 0] = 0

        s_vect_indices = np.where(self.alpha > 0 & (self.alpha < self.c))[0]
        self.bias = np.mean(y[s_vect_indices] - np.dot((self.alpha * y), self.kernel(x, x[s_vect_indices])))

    def decision_function(self, x):
        return np.dot((self.alpha * self.training_y), self.kernel(self.training_x, x)) + self.bias

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def label(self, x):
        return np.sign(self.predict(x))

    def score(self, x, y):
        return np.mean(np.sign(y) == self.label(x))

    def hinge_loss(self):
        ...


class DualMnistSVM(DualSVM):
    def __init__(self, **kwargs):
        self.target_num = kwargs.pop("num_to_recognize")
        super().__init__(training_labels=self.process_labels(kwargs.pop("training_labels")),
                         **kwargs)

    def process_labels(self, y):
        """Transforms the decimal labels into binary ones (it either looks for the number or does not)"""
        return np.where(y == self.target_num, 1, -1)

    def score(self, x, y):
        return super().score(x, self.process_labels(y))
