from tensorflow import keras 
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)

class CIFAR100Data(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        print('CIFAR100 Training data shape:', self.x_train.shape)
        print('CIFAR100 Training label shape', self.y_train.shape)
        print('CIFAR100 Test data shape', self.x_test.shape)
        print('CIFAR100 Test label shape', self.y_test.shape)

    def get_stretch_data(self, subtract_mean=True):
        """
        reshape X each image to row vector, and transform Y to one_hot label.
        :param subtract_mean:Indicate whether subtract mean image.
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = 100
        # x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float64')
        x_train = np.reshape(self.x_train, (self.x_train.shape[0], -1)).astype('float16')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        # x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float64')
        x_test = np.reshape(self.x_test, (self.x_test.shape[0], -1)).astype('float16')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0).astype('uint8')
            x_train -= mean_image
            x_test -= mean_image
            # print(x_mean[:10])
            # plt.figure(figsize=(4, 4))
            # plt.imshow(x_mean.reshape((32, 32, 3)))
            # plt.show()

        return x_train, y_train, x_test, y_test

    def get_data(self, subtract_mean=True, output_shape=None):
        """
        The data is not reshaped, keep 3 channel.
        :param subtract_mean:Indicate whether subtract mean image.
        :param output_shape:Indicate whether resize image
        :return: x_train, one_hot_y_train, x_test, one_hot_y_test
        """
        num_classes = 100
        x_train = self.x_train
        x_test = self.x_test
        # if output_shape:resize
        #     x_train = np.array([cv2.resize(img, output_shape) for img in self.x_train])
        #     x_test = np.array([cv2.(img, output_shape) for img in self.x_test])

        x_train = x_train.astype('float16')
        y_train = keras.utils.to_categorical(self.y_train, num_classes)

        x_test = x_test.astype('float16')
        y_test = keras.utils.to_categorical(self.y_test, num_classes)

        if subtract_mean:
            mean_image = np.mean(x_train, axis=0)
            x_train -= mean_image
            x_test -= mean_image
        return x_train, y_train, x_test, y_test


def plot_cifar100(cifar_data, num_sample_per_class):
    """
    random select num_sample_per_class to plot
    """
    num_classes = len(cifar_data.classes)

    plt.figure()
    for y, cls in enumerate(cifar_data.classes):
        cls_indices = np.flatnonzero(cifar_data.y_train == y)
        samples_indices = np.random.choice(cls_indices, num_sample_per_class, replace=False)
        samples = cifar_data.x_train[samples_indices]
        for x, sample in enumerate(samples):
            # subplot index count from 1
            plt_idx = x * num_classes + y + 1
            plt.subplot(num_sample_per_class, num_classes, plt_idx)
            plt.imshow(sample)
            plt.axis('off')
            if x == 0:
                plt.title(cls)
    plt.show()