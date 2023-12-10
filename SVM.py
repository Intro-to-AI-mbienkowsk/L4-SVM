import numpy as np
from util import *
import abc


class SVM:
    def __init__(self, num_to_recognize: int, lbda=DEFAULT_LAMBDA, learning_rate=DEFAULT_LEARNING_RATE,
                 batch_size=2 ** DEFAULT_BATCH_EXPONENT, epochs=DEFAULT_EPOCHS):
        self.num_to_recognize = num_to_recognize
        self.weights = np.zeros(784)
        self.lbda = lbda
        self.bias = 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, x, y):
        # if a number is not the one we're looking for, it is classified as -1
        bin_y = self.get_binary_y(y)
        for _ in range(self.epochs):
            for idx, x_i in enumerate(x):
                is_correct = bin_y[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if is_correct:
                    # shrink weight vector to maximize separation
                    self.weights -= self.learning_rate * (2 * self.lbda * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lbda * self.weights - np.dot(x_i, bin_y[idx]))
                    self.bias -= self.learning_rate * bin_y[idx]

    def hinge_loss(self, x, y):
        bin_y = self.get_binary_y(y)
        distances = 1 - bin_y * (np.dot(x, self.weights) - self.bias)
        losses = np.maximum(0, distances)
        regularization = self.lbda * REGULARIZATION_SCALAR * np.dot(self.weights, self.weights)
        mean_loss = np.mean(losses) + regularization
        return mean_loss

    def predict(self, x):
        return np.sign(np.dot(x, self.weights) + self.bias)

    def get_binary_y(self, y):
        """Maps the given label array onto a binary one containing 1 if the number is equal
        to the number we're trying to classify, else -1
        """
        return np.where(y != self.num_to_recognize, -1, 1)
