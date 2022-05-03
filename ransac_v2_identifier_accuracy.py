# -*- coding: utf-8 -*-
"""RANSAC_v2 Identifier Accuracy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r_F2EOa0JUUIxhOpom2NkIscwXnrWzQl
"""

from scipy.stats import entropy
from tensorflow.keras.datasets import cifar10
import numpy as np
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import os
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
        while label == trainY[trainYSwitchIndexes[i]]:
            label = random.choice(range(10))
        # switch label
        copyTrainY[trainYSwitchIndexes[i]] = label

    return copyTrainY


def splitTrainingData(trainX, trainY, splitPercentage):
    # get number of elements to split
    numberSplit = int(splitPercentage * len(trainX))
    # generate indexes to split
    indexes = list(range(len(trainX)))
    beforeSplitIndexes = random.sample(range(0, len(trainX)), numberSplit)
    afterSplitIndexes = list(set(indexes)-set(beforeSplitIndexes))

    # make new arrays
    firstTrainX = []
    firstTrainY = []
    secondTrainX = []
    secondTrainY = []

    # add each data sample to corresponding list
    for index in beforeSplitIndexes:
        firstTrainX.append(trainX[index])
        firstTrainY.append(trainY[index])
    for index in afterSplitIndexes:
        secondTrainX.append(trainX[index])
        secondTrainY.append(trainY[index])
    return np.array(firstTrainX), np.array(firstTrainY), np.array(secondTrainX), np.array(secondTrainY), beforeSplitIndexes, afterSplitIndexes


def trainModel(X, Y, n):
    # load pretrained model
    model = tf.keras.models.load_model('Pre Training/pretrain_model.h5')

    # model description
    # model.summary()

    # Compile
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit
    model.fit(X, Y, epochs=20)

    return model


def makeConfidentTrainingSets(model, firstTrainX, firstTrainY, secondTrainX, secondTrainY, entropyThreshold, peakThreshold, beforeSplitIndexes, afterSplitIndexes):
    newTrainX = []
    newTrainY = []
    confidentIndexes = []
    # find confident samples from first training set
    # obtain probability distribution of classes for each sample after the split and calculate its entropy
    # make predictions
    firstTrainXPredictions = model.predict(firstTrainX)
    # find entropy and peak value for every sample
    firstTrainXEntropies = []
    firstTrainXPeakValues = []
    for sample in firstTrainXPredictions:
        # calculate entropy
        sampleEntropy = entropy(sample)
        # calculate peak value
        probSorted = sorted(sample)
        probSorted = probSorted[::-1]
        peakValue = probSorted[0]/probSorted[1]

        if peakValue > 100:
            peakValue = 100

        firstTrainXEntropies.append(sampleEntropy)
        firstTrainXPeakValues.append(peakValue)

    # set NANs to 100
    firstTrainXPeakValues = np.array(firstTrainXPeakValues)
    firstTrainXPeakValues[np.isnan(firstTrainXPeakValues)] = 100

    # obtain samples that were correctly predicted and fall under the threshold for entropy and peak value
    for i in range(len(firstTrainXPredictions)):
        probDist = firstTrainXPredictions[i]
        predictedClass = np.argmax(probDist)

        # if confident add to list
        if predictedClass == firstTrainY[i] and firstTrainXEntropies[i] <= entropyThreshold and firstTrainXPeakValues[i] >= peakThreshold:
            newTrainX.append(firstTrainX[i])
            newTrainY.append(firstTrainY[i])
            confidentIndexes.append(beforeSplitIndexes[i])

    # find confident samples from unused training set
    # obtain probability distribution of classes for each sample after the split and calculate its entropy
    # make predictions
    secondTrainXPredictions = model.predict(secondTrainX)
    # find entropy and peak value for every sample
    secondTrainXEntropies = []
    secondTrainXPeakValues = []
    for sample in secondTrainXPredictions:
        # calculate entropy
        sampleEntropy = entropy(sample)
        # calculate peak value
        probSorted = sorted(sample)
        probSorted = probSorted[::-1]
        peakValue = probSorted[0]/probSorted[1]

        if peakValue > 100:
            peakValue = 100

        secondTrainXEntropies.append(sampleEntropy)
        secondTrainXPeakValues.append(peakValue)

    # set NANs to 0
    secondTrainXPeakValues = np.array(secondTrainXPeakValues)
    secondTrainXPeakValues[np.isnan(secondTrainXPeakValues)] = 100

    # obtain samples that were correctly predicted and fall under the threshold for entropy and peak value
    for i in range(len(secondTrainXPredictions)):
        probDist = secondTrainXPredictions[i]
        predictedClass = np.argmax(probDist)

        # if confident add to list
        if predictedClass == secondTrainY[i] and secondTrainXEntropies[i] <= entropyThreshold and secondTrainXPeakValues[i] >= peakThreshold:
            newTrainX.append(secondTrainX[i])
            newTrainY.append(secondTrainY[i])
            confidentIndexes.append(afterSplitIndexes[i])

    # # make plots

    print('saving images')

    firstPlotIndices = list(range(len(firstTrainXEntropies)))

    sortedFirstTrainXEntropies = sorted(firstTrainXEntropies)
    plt.plot(firstPlotIndices, sortedFirstTrainXEntropies)
    plt.savefig('firstSetEntropy.png')
    plt.close()

    sortedFirstTrainXPeaks = sorted(firstTrainXPeakValues)
    plt.plot(firstPlotIndices, sortedFirstTrainXPeaks)
    plt.savefig('firstSetPeak.png')
    plt.close()

    secondPlotIndices = list(range(len(secondTrainXEntropies)))

    sortedSecondTrainXEntropies = sorted(secondTrainXEntropies)
    plt.plot(secondPlotIndices, sortedSecondTrainXEntropies)
    plt.savefig('secondSetEntropy.png')
    plt.close()

    sortedSecondTrainXPeaks = sorted(secondTrainXPeakValues)
    plt.plot(secondPlotIndices, sortedSecondTrainXPeaks)
    plt.savefig('secondSetPeak.png')
    plt.close()

    return newTrainX, newTrainY, confidentIndexes


(trainX, trainY), (testX, testY) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
trainX, testX = trainX / 255.0, testX / 255.0

# flatten the label values
trainY, testY = trainY.flatten(), testY.flatten()

# corrupt data
noisePercentage = 0.25
trainYMislabeled = corruptData(trainY, noisePercentage)

# cleanModel = load_model('CleanModelTraining/ransac_clean.h5')
# upperBoundAccuracy = cleanModel.evaluate(testX, testY)[1]

# print(upperBoundAccuracy)

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# collect best indexes over multiple models
bestIndexes = list(range(len(trainX)))
for i in range(5):
    # split data
    splitPercentage = .7
    firstTrainX, firstTrainY, secondTrainX, secondTrainY, beforeSplitIndexes, afterSplitIndexes = splitTrainingData(
        trainX, trainYMislabeled, splitPercentage)

    # train model used to identify confident samples
    confidenceModel = trainModel(firstTrainX, firstTrainY, 1)
    entropyThresholds = [1]  # [0.25, .5, 1, 3]
    peakThresholds = [5]  # [5, 3, 1, .5]
    for j in range(len(entropyThresholds)):
        entropyThreshold = entropyThresholds[j]
        peakThreshold = peakThresholds[j]

        # find samples that this model is confident on
        newTrainX, newTrainY, confidentIndexes = makeConfidentTrainingSets(
            confidenceModel, firstTrainX, firstTrainY, secondTrainX, secondTrainY, entropyThreshold, peakThreshold, beforeSplitIndexes, afterSplitIndexes)

        # add 1 to every confident image
        for index in confidentIndexes:
            bestIndexes[index] += 1


# sort and preserve index
bestSorted = np.argsort(bestIndexes)
bestSorted = bestSorted[::-1]

percs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
accuracies = []

for perc in percs:
    print('Using', perc, 'of confident samples')
    # calculate number of samples to use
    numberCertain = int(perc * len(bestIndexes))

    # make new datasets for the most confident samples
    bestTrainX = []
    bestTrainY = []

    # take certain samples
    for i in range(numberCertain):
        bestTrainX.append(trainX[bestSorted[i]])
        bestTrainY.append(trainYMislabeled[bestSorted[i]])

    # run experiments
    # train a new model on these confident samples
    bestTrainX = np.array(bestTrainX)
    bestTrainY = np.array(bestTrainY)
    ransacModel = trainModel(bestTrainX, bestTrainY, 1)

    # calculate accuracy of this model in using test data
    accuracy = ransacModel.evaluate(testX, testY)[1]

    accuracies.append(accuracy)

    print('This model has an accuracy of', accuracy, 'on the testing data.')

print("accuracies:", accuracies)
plt.plot(percs, accuracies)
plt.savefig('accuracyChar.png')
plt.close()
