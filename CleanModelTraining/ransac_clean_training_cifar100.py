# -*- coding: utf-8 -*-
"""RANSAC_CLEAN_TRAINING.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tFoS_uMOG6M3HA8fpIhSxCKc6QcIF9KK
"""
import sys
sys.path.append('../')
import os
from tensorflow.keras.datasets import cifar100
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from cifar100_clean_utils import *
from tensorflow.keras import losses
from ResNet import ResNet32ForCIFAR10
from tensorflow.keras.callbacks import LearningRateScheduler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)

# get data
cifar100_data = CIFAR100Data()
trainX, trainY, testX, testY = cifar100_data.get_data(subtract_mean=True)

weight_decay = 1e-4
lr = 1e-1
num_classes = 100
cleanModel = ResNet32ForCIFAR10(input_shape=(
    32, 32, 3), classes=num_classes, weight_decay=weight_decay)
opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
cleanModel.compile(optimizer=opt,
                   loss=losses.categorical_crossentropy,
                   metrics=['accuracy'])
# cleanModel.summary()


def lr_scheduler(epoch):
    new_lr = lr
    if epoch <= 21:
        pass
    elif epoch > 21 and epoch <= 37:
        new_lr = lr * 0.1
    else:
        new_lr = lr * 0.01
    print('new lr:%.2e' % new_lr)
    return new_lr


reduce_lr = LearningRateScheduler(lr_scheduler)

# # load pretrained model
# cleanModel = tf.keras.models.load_model(
#     '../Pre Training/cifar100_pretrain_model.h5')

# # model description
# # model.summary()

# # Compile
# lr = 1e-1
# opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)

# cleanModel.compile(
#     optimizer=opt, loss=losses.categorical_crossentropy, metrics=['accuracy'])

# Fit
r = cleanModel.fit(trainX, trainY, epochs=50,
                   batch_size=128, callbacks=[reduce_lr])

# obtain results
upperBoundAccuracy = cleanModel.evaluate(testX, testY)[1]

print('The clean model has an accuracy of',
      upperBoundAccuracy, 'on the testing data.')

# #save model
# cleanModel.save('ransac_clean.h5')
