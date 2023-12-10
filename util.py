from mnist import *
from enum import Enum


class TRAINING_METHOD(Enum):
    SGD = 1
    BGD = 2


def import_data():
    train_x = train_images()
    train_y = train_labels()
    test_x = test_images()
    test_y = test_labels()
    train_x_processed = train_x.reshape(train_x.shape[0], -1) / 255.0
    test_x_processed = test_x.reshape(test_x.shape[0], -1) / 255.0
    return (train_x_processed, train_y), (test_x_processed, test_y)


DEFAULT_LEARNING_RATE = .0001
DEFAULT_LAMBDA = 1
DEFAULT_BATCH_EXPONENT = 6
REGULARIZATION_SCALAR = .5
DEFAULT_EPOCHS = 100
