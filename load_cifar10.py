# load_cifar10.py

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Save preprocessed data if needed
# np.save('x_train.npy', x_train)
# np.save('y_train.npy', y_train)
# np.save('x_test.npy', x_test)
# np.save('y_test.npy', y_test)
