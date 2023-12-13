import numpy as np
from util import *


class PrimalSVM:
    def __init__(self, training_data, training_labels, c=DEFAULT_C, learning_rate=DEFAULT_LEARNING_RATE, dimension=784,
                 epochs=DEFAULT_EPOCHS, training_method=TRAINING_METHOD.SGD):
        self.weights = np.zeros(dimension)
        self.c = c
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_method = training_method
        self.dimension = dimension
        self.losses = []
        self.train_x = training_data
        self.train_y = training_labels

    def fit(self):
        x, y = self.train_x, self.train_y
        for i in range(self.epochs):
            print(f'epoch {i}')
            if self.training_method == TRAINING_METHOD.SGD:
                margin = y * self.decision_function(x)
                missclassified_indices = np.where(margin < 1)[0]

                grad_w = self.weights - self.c * y[missclassified_indices].dot(x[missclassified_indices])
                self.weights = self.weights - grad_w * self.learning_rate

                grad_b = self.c * np.sum(y[missclassified_indices])
                self.bias = self.bias - grad_b * self.learning_rate

            self.losses.append(self.hinge_loss())

    def hinge_loss(self):
        return (np.dot(self.weights, self.weights) + self.c * np.sum(
            np.maximum(0, 1 - self.train_y * self.decision_function(self.train_x)))) / 2

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def decision_function(self, x):
        return np.dot(x, self.weights) + self.bias

    def score(self, test_x, test_y):
        return np.mean(np.sign(test_y) == self.predict(test_x))


class MnistPrimalSVM(PrimalSVM):
    def __init__(self, *args, **kwargs):
        self.target_num = kwargs.pop('num_to_recognize')
        train_y = kwargs.pop('training_data')
        super().__init__(*args, dimension=784, training_data=self.binary_labels(train_y), **kwargs)

    def binary_labels(self, y):
        """Maps the given label array onto a binary one containing 1 if the number is equal
        to the number we're trying to classify, else -1
        """
        return np.where(y != self.target_num, -1, 1)
