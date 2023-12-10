from mnist import *


def import_data():
    train_x = train_images()
    train_y = train_labels()
    test_x = test_images()
    test_y = test_labels()
    train_x_flattened = train_x.reshape(train_x.shape[0], -1)
    test_x_flattened = test_x.reshape(test_x.shape[0], -1)
    return (train_x_flattened, train_y), (test_x_flattened, test_y)


DEFAULT_LEARNING_RATE = .001
DEFAULT_LAMBDA = 1
DEFAULT_BATCH_EXPONENT=6
REGULARIZATION_SCALAR=.5
DEFAULT_EPOCHS=1000