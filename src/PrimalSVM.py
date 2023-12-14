import matplotlib.pyplot as plt
import numpy as np
from src.util import *


class PrimalSVM:
    def __init__(self, training_data, training_labels, c=DEFAULT_PRIMAL_C, learning_rate=DEFAULT_PRIMAL_LEARNING_RATE,
                 dimension=784,
                 epochs=DEFAULT_EPOCHS):
        self.weights = np.zeros(dimension)
        self.c = c
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dimension = dimension
        self.losses = []
        self.train_x = training_data
        self.train_y = training_labels

    def fit(self):
        x, y = self.train_x, self.train_y
        best_weights = self.weights
        best_bias = self.bias
        best_loss = float('inf')

        for i in range(self.epochs):
            margin = y * self.decision_function(x)
            missclassified_indices = np.where(margin < 1)[0]

            dw = self.weights - self.c * (x[missclassified_indices].T * y[missclassified_indices]).T.sum(axis=0)
            db = - self.c * np.sum(y[missclassified_indices])

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self.hinge_loss()
            self.losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_bias = self.bias
                best_weights = self.weights

            print(
                f"epoch {i}, loss {self.losses[-1]}, dw={round(np.linalg.norm(dw), 2)}, db={round(np.linalg.norm(db), 2)}")
        self.weights = best_weights
        self.bias = best_bias

    def hinge_loss(self):
        return (np.dot(self.weights / 2, self.weights) + self.c * np.sum(
            np.maximum(0, 1 - self.train_y * self.decision_function(self.train_x))))

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def decision_function(self, x):
        return np.dot(x, self.weights) + self.bias

    def score(self, test_x, test_y):
        return np.mean(np.sign(test_y) == self.predict(test_x))


class MnistPrimalSVM(PrimalSVM):
    def __init__(self, *args, **kwargs):
        self.target_num = kwargs.pop('num_to_recognize')
        train_y = kwargs.pop('training_labels')
        super().__init__(*args, dimension=784, training_labels=self.process_labels(train_y), **kwargs)

    def process_labels(self, y):
        """Maps the given label array onto a binary one containing 1 if the number is equal
        to the number we're trying to classify, else -1
        """
        return np.where(y != self.target_num, -1, 1)

    def plot_losses(self):
        fig = plt.figure()
        fig.set_facecolor("lightblue")
        xlabels = np.arange(len(self.losses))
        plt.plot(xlabels, self.losses, color='#2596be')
        plt.title(f"Hinge loss in each epoch, searching for {self.target_num}", weight="bold")
        plt.xlabel("epoch nr")
        plt.ylabel("loss")
        plt.grid(True)
        plt.show()
