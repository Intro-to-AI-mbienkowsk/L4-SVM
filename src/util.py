from mnist import *
from enum import Enum

DEFAULT_PRIMAL_LEARNING_RATE = .00001
DEFAULT_EPOCHS = 1000
DEFAULT_PRIMAL_C = 15

DEFAULT_DUAL_C = 10
DEFAULT_DUAL_LEARNING_RATE = .0001
DEFAULT_SIGMA = .01
DEFAULT_DEGREE = 2
POLYNOMIAL_KERNEL_CONSTANT_TERM = 1


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
