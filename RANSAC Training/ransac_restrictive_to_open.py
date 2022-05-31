# -*- coding: utf-8 -*-
"""RANSAC_v2 Identifier Accuracy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r_F2EOa0JUUIxhOpom2NkIscwXnrWzQl
"""
import sys
sys.path.append('../')
import itertools
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import losses
from ResNet import ResNet20ForCIFAR10
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import numpy as np
from scipy.stats import entropy
from cifar10_ransac_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # (xxxx is your specific GPU ID)

# method to add noisy labels to data


def corruptData(trainY, noisePercentage):
    # create copies of labels
    copyTrainY = trainY.copy()

    # calculate number of samples to be made noisy
    numberNoisyTrain = int(noisePercentage * len(copyTrainY))

    # generate indexes to swap
    trainYSwitchIndexes = random.sample(
        range(0, len(copyTrainY)), numberNoisyTrain)

    # generate new classes not equal to original for training and switch class
    for i in range(len(trainYSwitchIndexes)):
        label = random.choice(range(10))
        # find label that isn't the same
        while label == np.argmax(trainY[trainYSwitchIndexes[i]]):
            label = random.choice(range(10))
        # switch label
        newLabel = np.zeros(10)
        newLabel[label] = 1
        copyTrainY[trainYSwitchIndexes[i]] = np.array(newLabel)

    return copyTrainY


def trainModel(X, Y):
    # compile a model
    weight_decay = 1e-4
    lr = 1e-1
    num_classes = 10
    model = ResNet20ForCIFAR10(input_shape=(
        32, 32, 3), classes=num_classes, weight_decay=weight_decay)
    opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
    model.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

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

    # fit model
    model.fit(X, Y, epochs=50,
              batch_size=128, callbacks=[reduce_lr])

    return model


def makeConfidentTrainingSets(model, corTrainX, corTrainY, entropyThreshold, peakThreshold):
    newTrainX = []
    newTrainY = []
    # find confident samples from first training set
    # obtain probability distribution of classes for each sample after the split and calculate its entropy
    # make predictions
    entropies = []
    peaks = []

    predictions = model.predict(corTrainX)
    # find entropy for every sample and decide if confident
    for i in range(len(predictions)):
        sample = predictions[i]
        # get classification
        predictedClass = np.argmax(sample)
        # calculate entropy
        sampleEntropy = entropy(sample)

        # calculate peak value
        probSorted = sorted(sample)
        probSorted = probSorted[::-1]
        # sum all prob except max
        probSum = 0
        for j in range(1, len(probSorted)):
            probSum += probSorted[j]
        peakValue = probSorted[0]/probSum

        if np.isnan(peakValue) or peakValue > 1000:
            peakValue = 1000

        # if confident add to list
        if predictedClass == np.argmax(corTrainY[i]) and sampleEntropy <= entropyThreshold and peakValue >= peakThreshold:
            newTrainX.append(corTrainX[i])
            newTrainY.append(corTrainY[i])

        entropies.append(sampleEntropy)
        peaks.append(peakValue)

    print('saving images')

    indices = list(range(len(corTrainX)))

    sortedEntropies = sorted(entropies)
    plt.plot(indices, sortedEntropies)
    plt.savefig('entropyvalues.png')
    plt.close()

    sortedPeaks = sorted(peaks)
    plt.plot(indices, sortedPeaks)
    plt.savefig('peakvalues.png')
    plt.close()

    return np.array(newTrainX), np.array(newTrainY)


# get data
cifar10_data = CIFAR10Data()
trainX, trainY, testX, testY = cifar10_data.get_data(subtract_mean=True)

# corrupt data
noisePercentage = 0.1
trainYMislabeled = corruptData(trainY, noisePercentage)

# print(upperBoundAccuracy)

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# set thresholds
entropyThresholds = [.1, .2, .3, .4, .5]
peakThresholds = [400, 200, 100, 50, 25]

# current training set
currentTrainX = trainX
currentTrainY = trainYMislabeled

for p in range(5):
    # train model used to identify confident samples
    confidenceModel = trainModel(currentTrainX, currentTrainY)
    # from cross validation
    entropyThreshold = entropyThresholds[p]
    peakThreshold = peakThresholds[p]

    # find samples that this model is confident on
    newTrainX, newTrainY = makeConfidentTrainingSets(
        confidenceModel, currentTrainX, currentTrainY, entropyThreshold, peakThreshold)

    # set current training sets
    currentTrainX = newTrainX
    currentTrainY = newTrainY

# #count how accurate model was
# correctLabelInCertain = 0
# mislabelInCertain = 0
# for i in range(currentTrainX):
#     if np.argmax(trainYMislabeled[bestSorted[i]]) == np.argmax(trainY[bestSorted[i]]):
#         correctLabelInCertain += 1
#     else:
#         mislabelInCertain += 1


# run experiments
# train a new model on these confident samples
ransacModel = trainModel(currentTrainX, currentTrainY)

# calculate accuracy of this model in using test data
accuracy = ransacModel.evaluate(testX, testY)[1]

print('This model has an accuracy of', accuracy, 'on the testing data.\n')
