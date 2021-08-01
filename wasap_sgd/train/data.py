### Data class and associated helper methods

import numpy as np
import logging


class Data(object):
    """Class providing an interface to the input training and testing data.
        Attributes:
          x_train: array of data points to use for training
          y_train: array of labels to use for training
          x_test: array of data points to use for testing
          y_test: array of labels to use for testing
          batch_size: size of training batches
    """

    def __init__(self, x_train, y_train, x_test, y_test, batch_size, augmentation=False, dataset='cifar10'):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.augmentation = (augmentation and dataset == 'cifar10')
        self.dataset = dataset

        if self.augmentation:
            from keras.preprocessing.image import ImageDataGenerator
            self.datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            self.datagen.fit(self.x_train)
        else:
            self.datagen = None

    def generate_data(self):
        while True:
            self.shuffle()
            try:
                if self.augmentation:
                    for B in self.generate_augmented_data():
                        yield B
                else:
                    for B in self.generate_train_data():
                        yield B
            except StopIteration:
                logging.warning("start over generator loop")

    def generate_augmented_data(self):
        """Yields batches of augmented training data until none are left."""
        output_generator = self.datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
        for j in range(self.x_train.shape[0] // self.batch_size):
            x_b, y_b = next(output_generator)
            x_b = x_b.reshape(-1, 32 * 32 * 3)

            yield x_b, y_b

    def generate_test_data(self):
        """Yields batches of training data until none are left."""
        for j in range(self.x_test.shape[0] // self.batch_size):
            start_pos = j * self.batch_size
            end_pos = (j + 1) * self.batch_size

            yield self.x_test[start_pos:end_pos], self.y_test[start_pos:end_pos]

    def generate_train_data(self):
        """Yields batches of training data until none are left."""
        for j in range(self.x_train.shape[0] // self.batch_size):
            start_pos = j * self.batch_size
            end_pos = (j + 1) * self.batch_size

            yield self.x_train[start_pos:end_pos], self.y_train[start_pos:end_pos]

    def count_data(self):
        return self.x_train.shape[0]

    def shuffle(self):
        seed = np.arange(self.x_train.shape[0])
        np.random.shuffle(seed)
        self.x_train = self.x_train[seed]
        self.y_train = self.y_train[seed]

    def is_numpy_array(self, data):
        return isinstance(data, np.ndarray)

    def get_train_data(self):
        if self.augmentation:
            return self.x_train.reshape(-1, 32 * 32 * 3)
        else:
            return self.x_train

    def get_test_data(self):
        if self.augmentation and self.dataset == 'cifar10':
            return self.x_test.reshape(-1, 32 * 32 * 3)
        else:
            return self.x_test

    def get_train_labels(self):
        return self.y_train

    def get_test_labels(self):
        return self.y_test

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
            Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])
