from mnist import *
from enum import Enum


class TRAINING_METHOD(Enum):
    SGD = 1
    BGD = 2


class KERNEL(Enum):
    POLYNOMIAL = 1
    GAUSSIAN = 2


def import_data():
    train_x = train_images()
    train_y = train_labels()
    test_x = test_images()
    test_y = test_labels()
    train_x_processed = train_x.reshape(train_x.shape[0], -1) / 255.0
    test_x_processed = test_x.reshape(test_x.shape[0], -1) / 255.0
    return (train_x_processed, train_y), (test_x_processed, test_y)


DEFAULT_LEARNING_RATE = .001
DEFAULT_EPOCHS = 10
DEFAULT_DEGREE = 2
DEFAULT_SIGMA = .05
DEFAULT_NUM_TO_RECOGNIZE = 0
DEFAULT_C = 15
POLYNOMIAL_KERNEL_CONSTANT_TERM = 1
