from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from DualSVM import DualMnistSVM
from PrimalSVM import MnistPrimalSVM
from util import import_data
import numpy as np


class MnistClassifier:
    def __init__(self, primal=False):

        (train_x, train_y), (test_x, test_y) = import_data()
        self.test_x = test_x[:1000]
        self.test_y = test_y[:1000]
        self.predictions = None

        if primal:
            self.svms = [MnistPrimalSVM(training_data=train_x, training_labels=train_y, c=15, num_to_recognize=i) for i in range(10)]

        else:
            self.svms = [DualMnistSVM(num_to_recognize=i, training_data=train_x[:5000], training_labels=train_y[:5000])
                         for i in range(10)]

    def predict(self, x):
        decisions = np.array([svm.decision_function(x) for svm in self.svms])
        max_decision_indices = np.argmax(decisions, axis=0)
        num_values = [self.svms[i].target_num for i in max_decision_indices]
        return num_values

    def score(self):
        return np.mean(self.predictions == self.test_y)

    def train(self):
        for svm in self.svms:
            print(f"SVM {self.svms.index(svm) + 1} training")
            svm.fit()
        self.predictions = self.predict(self.test_x)

    def display_confusion_matrix(self):
        cm = confusion_matrix(self.test_y, np.sum(self.predictions, axis=0))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
