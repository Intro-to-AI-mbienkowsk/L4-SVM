import matplotlib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
from DualSVM import DualMnistSVM
from PrimalSVM import MnistPrimalSVM
from util import import_data, DEFAULT_C
import numpy as np

matplotlib.use('qtagg')


class MnistClassifier:
    def __init__(self, primal=False):

        (train_x, train_y), (test_x, test_y) = import_data()
        self.test_x = test_x if primal else test_x[:1000]
        self.test_y = test_y if primal else test_y[:1000]
        self.predictions = None

        if primal:
            self.svms = [
                            MnistPrimalSVM(training_data=train_x, training_labels=train_y, num_to_recognize=i) for i
                            in range(9)
                        ] + [MnistPrimalSVM(training_data=train_x, training_labels=train_y, num_to_recognize=i,
                                            epochs=2000, learning_rate=1e-7) for i in
                             range(8, 10)]

        else:
            self.svms = [DualMnistSVM(num_to_recognize=i, training_data=train_x[:5000], training_labels=train_y[:5000])
                         for i in range(10)]

    def predict(self, x):
        decisions = np.array([svm.decision_function(x) for svm in self.svms])
        output = np.full(decisions.shape[1], -1)
        max_decisions = np.max(decisions, axis=0)
        max_decision_indices = np.argmax(decisions, axis=0)
        for i, max_decision in enumerate(max_decisions):
            if max_decision > 0:
                output[i] = self.svms[
                    max_decision_indices[i]].target_num
        return output

    def score(self):
        return np.mean(self.predictions == self.test_y)

    def train(self):
        for svm in self.svms:
            print(f"SVM {self.svms.index(svm) + 1} training")
            svm.fit()
        self.predictions = self.predict(self.test_x)

    def display_confusion_matrix(self):
        labels = [i for i in range(-1, 10)]
        cm = confusion_matrix(self.test_y, self.predictions, labels=labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        precision = precision_score(self.test_y, self.predictions, average='weighted',
                                    labels=labels[1:])
        accuracy = accuracy_score(self.test_y, self.predictions)
        f1 = f1_score(self.test_y, self.predictions, average='weighted', labels=labels[1:])

        textstr = '\n'.join((
            f'Precision: {precision:.2f}',
            f'Accuracy: {accuracy:.2f}',
            f'F1 Score: {f1:.2f}'))
        plt.subplots_adjust(right=.7)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gcf().text(0.73, 0.5, textstr, fontsize=14, bbox=props)

        plt.show()
