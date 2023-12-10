from SVM import SVM, MnistSVM
from util import import_data


def main():
    svm = MnistSVM(num_to_recognize=1, epochs=25, learning_rate=0.0001)
    (train_x, train_y), (test_x, test_y) = import_data()
    svm.fit(train_x, train_y)
    score = svm.score(test_x, test_y)
    print(f"Accuracy: {round(score, 2)}")


if __name__ == '__main__':
    main()