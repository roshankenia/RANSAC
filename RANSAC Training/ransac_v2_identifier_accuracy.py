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


def makeConfidentTrainingSets(model, corTrainX, corTrainY, entropyThreshold, peakThreshold, trainY):
    newTrainX = []
    newTrainY = []
    confidentIndexes = []
    # find confident samples from first training set
    # obtain probability distribution of classes for each sample after the split and calculate its entropy
    # make predictions
    entropies = []
    peaks = []
    groundTruthDouble = 0
    nonGroundTruth = 0
    groundTruthOnly = 0

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
            # only add if correct class as well
            if predictedClass == np.argmax(trainY[i]):
                newTrainX.append(corTrainX[i])
                newTrainY.append(corTrainY[i])
                confidentIndexes.append(i)
                groundTruthDouble += 1
            else:
                newTrainX.append(corTrainX[i])
                newTrainY.append(corTrainY[i])
                confidentIndexes.append(i)
                nonGroundTruth += 1
        # check if model predicted correct class but is mislabeled data
        elif predictedClass != np.argmax(corTrainY[i]) and predictedClass == np.argmax(trainY[i]):
            newTrainX.append(corTrainX[i])
            newTrainY.append(corTrainY[i])
            confidentIndexes.append(i)
            groundTruthOnly += 1

        entropies.append(sampleEntropy)
        peaks.append(peakValue)

    print('Label match both:', groundTruthDouble)
    print('Label match only training set:', nonGroundTruth)
    print('Label match only ground truth:', groundTruthOnly)
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

    return np.array(newTrainX), np.array(newTrainY), confidentIndexes


# get data
cifar10_data = CIFAR10Data()
trainX, trainY, testX, testY = cifar10_data.get_data(subtract_mean=True)

# corrupt data
noisePercentage = 0.1
trainYMislabeled = corruptData(trainY, noisePercentage)

# print(upperBoundAccuracy)

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# collect best indexes over multiple models
bestIndexes = list(itertools.repeat(0, len(trainX)))
for p in range(5):
    # select subset of data to train on
    # calculate number of samples to be added to subset
    numberTrain = int(0.75 * len(trainX))

    # generate indexes to use
    trainIndexes = random.sample(
        range(0, len(trainX)), numberTrain)

    # add subset samples to correct arrays
    subsetTrainX = []
    subsetTrainY = []
    for index in trainIndexes:
        subsetTrainX.append(trainX[index])
        subsetTrainY.append(trainYMislabeled[index])
    subsetTrainX = np.array(subsetTrainX)
    subsetTrainY = np.array(subsetTrainY)

    # train model used to identify confident samples
    confidenceModel = trainModel(subsetTrainX, subsetTrainY)
    # from cross validation
    entropyThreshold = .1
    peakThreshold = 400

    # find samples that this model is confident on
    newTrainX, newTrainY, confidentIndexes = makeConfidentTrainingSets(
        confidenceModel, trainX, trainYMislabeled, entropyThreshold, peakThreshold, trainY)

    # add 1 to every confident image
    for index in confidentIndexes:
        bestIndexes[index] += 1


# sort and preserve index
bestSorted = np.argsort(bestIndexes)
bestSorted = bestSorted[::-1]

print('best 50 samples:')
for g in range(50):
    print(bestSorted[g], ':', bestIndexes[bestSorted[g]])

percs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
accuracies = []
correctLabelInCertains = []
mislabelInCertains = []
correctLabelInUncertains = []
mislabelInUncertains = []

for perc in percs:
    print('Using', perc, 'of confident samples')
    # calculate number of samples to use
    numberCertain = int(perc * len(bestIndexes))

    # make new datasets for the most confident samples
    bestTrainX = []
    bestTrainY = []

    correctLabelInCertain = 0
    mislabelInCertain = 0
    correctLabelInUncertain = 0
    mislabelInUncertain = 0

    # take certain samples and count those that were correctly and incorrectly identified
    for i in range(numberCertain):
        bestTrainX.append(trainX[bestSorted[i]])
        bestTrainY.append(trainYMislabeled[bestSorted[i]])

        if np.argmax(trainYMislabeled[bestSorted[i]]) == np.argmax(trainY[bestSorted[i]]):
            correctLabelInCertain += 1
        else:
            mislabelInCertain += 1

    for j in range(numberCertain, len(bestSorted)):
        if np.argmax(trainYMislabeled[bestSorted[j]]) == np.argmax(trainY[bestSorted[j]]):
            correctLabelInUncertain += 1
        else:
            mislabelInUncertain += 1

    correctLabelInCertains.append(correctLabelInCertain)
    mislabelInCertains.append(mislabelInCertain)
    correctLabelInUncertains.append(correctLabelInUncertain)
    mislabelInUncertains.append(mislabelInUncertain)

    # run experiments
    # train a new model on these confident samples
    bestTrainX = np.array(bestTrainX)
    bestTrainY = np.array(bestTrainY)
    ransacModel = trainModel(bestTrainX, bestTrainY)

    # calculate accuracy of this model in using test data
    accuracy = ransacModel.evaluate(testX, testY)[1]

    accuracies.append(accuracy)

    print('This model has an accuracy of', accuracy, 'on the testing data.\n')
    print('This model had', correctLabelInCertain,
          'correctly labeled samples in the certain training data out of', len(bestTrainY), '\n')
    print('This model had', mislabelInCertain,
          'mislabeled labeled samples in the certain training data out of', len(bestTrainY), '\n')
    print('This model had', correctLabelInUncertain,
          'correctly labeled samples in the certain training data out of', (len(bestIndexes) - len(bestTrainY)), '\n')
    print('This model had', mislabelInUncertain,
          'mislabeled labeled samples in the certain training data out of', (len(bestIndexes) - len(bestTrainY)), '\n')


print("accuracies:", accuracies)
plt.plot(percs, accuracies)
plt.savefig('accuracyChar.png')
plt.close()

print("correctLabelInCertains:", correctLabelInCertains)
plt.plot(percs, correctLabelInCertains)
plt.savefig('correctLabelInCertains.png')
plt.close()

print("mislabelInCertains:", mislabelInCertains)
plt.plot(percs, mislabelInCertains)
plt.savefig('mislabelInCertains.png')
plt.close()

print("correctLabelInUncertains:", correctLabelInUncertains)
plt.plot(percs, correctLabelInUncertains)
plt.savefig('correctLabelInUncertains.png')
plt.close()

print("mislabelInUncertains:", mislabelInUncertains)
plt.plot(percs, mislabelInUncertains)
plt.savefig('mislabelInUncertains.png')
plt.close()
