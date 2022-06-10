# -*- coding: utf-8 -*-
"""RANSAC_v2 Identifier Accuracy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r_F2EOa0JUUIxhOpom2NkIscwXnrWzQl
"""
import sys
sys.path.append('../')
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from cifar10_ransac_utils import *
from scipy.stats import entropy
import numpy as np
import random
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from ResNet import ResNet20ForCIFAR10
from tensorflow.keras import losses
from tensorflow.keras.callbacks import LearningRateScheduler
import itertools

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
    sampleArray = []
    # find confident samples from first training set
    # obtain probability distribution of classes for each sample after the split and calculate its entropy
    # make predictions
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

        if np.isnan(peakValue) or peakValue > 100:
            peakValue = 101

        confident = 0
        if predictedClass == np.argmax(corTrainY[i]) and sampleEntropy <= entropyThreshold and peakValue >= peakThreshold:
            confident = 1

        # determine how accurate classification was

        classificationScore = 0

        if predictedClass != np.argmax(corTrainY[i]) and predictedClass != np.argmax(trainY[i]):
            classificationScore = 0
        elif predictedClass == np.argmax(corTrainY[i]) and predictedClass != np.argmax(trainY[i]):
            classificationScore = 1
        elif predictedClass != np.argmax(corTrainY[i]) and predictedClass == np.argmax(trainY[i]):
            classificationScore = 2
        elif predictedClass == np.argmax(corTrainY[i]) and predictedClass == np.argmax(trainY[i]):
            classificationScore = 3

        sampleData = [predictedClass, sampleEntropy,
                      peakValue, confident, classificationScore]
        sampleArray.append(sampleData)

    return sampleArray


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
featureVector = []
for p in range(2):
    # select subset of data to train on
    # calculate number of samples to be added to subset
    numberTrain = int(1 * len(trainX))

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
    peakThreshold = 100

    # find samples that this model is confident on
    sampleArray = makeConfidentTrainingSets(
        confidenceModel, trainX, trainYMislabeled, entropyThreshold, peakThreshold, trainY)

    # add iteration data to feature vector
    featureVector.append(sampleArray)

# we first want to visualize the feature vector over the space

# lets first create a tensor for each sample over the iterations
# this vector will contain the stats for each sample
statVector = []
noiseVector = []
# iterate through each samples iteration data

for i in range(len(trainX)):
    # get iteration data
    iterData = []
    for iter in featureVector:
        iterData.append(iter[i])

    # keep track of the entropy and the peak value for the samples over each iteration
    entVals = []
    peakVals = []
    confidence = 0
    curLabel = iterData[0][0]
    consistent = 1
    predLabels = [0,0,0,0,0,0,0,0,0,0]
    for it in iterData:
        entVals.append(it[1])
        peakVals.append(it[2])
        predLabels[it[0]] += 1
        confidence += it[3]

        if it[0] != curLabel:
            consistent = 0

    # see if sample was confident in majority of predictions
    confident = 0
    if confidence > (len(featureVector)/2):
        confident = 1

    # calculate avg entropy and peak
    avgEnt = np.average(entVals)
    avgPeak = np.average(peakVals)

    # calculate variance for entropy and peak
    stdEnt = np.std(entVals)
    stdPeak = np.std(peakVals)

    #give an ensemble label
    ensembleLabel = np.argmax(predLabels)

    #temporary noisy label
    noisy = 0
    if np.argmax(trainY[i]) == np.argmax(trainYMislabeled[i]):
        noisy = 1

    # add data to stat vector
    data = [avgEnt, avgPeak, stdEnt, stdPeak, confident, consistent, noisy]
    statVector.append(data)

    # # lets try using the raw data itself
    # data = entVals + peakVals
    # statVector.append(data)

    # decide whether this was noisy data or not
    if np.argmax(trainY[i]) == np.argmax(trainYMislabeled[i]):
        noiseVector.append(1)
    else:
        noiseVector.append(0)

statVector = np.array(statVector)
noiseVector = np.array(noiseVector)

# We want to get TSNE embedding with 2 dimensions
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(statVector)
tsne_result.shape
# Two dimensions for each of our images
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame(
    {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1], 'label': noiseVector})
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                hue='label', data=tsne_result_df, ax=ax, s=10)
lim = (tsne_result.min()-5, tsne_result.max()+5)
plt.title('Average and Variance of Entropy and Peak Value Reduced')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.savefig('tSNE-Results.png')
plt.close()

# lets also do a graph of just entropy vs peak value
result_df = pd.DataFrame(
    {'Entropy': statVector[:, 0], 'Peak Value': statVector[:, 1], 'label': noiseVector})
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='Entropy', y='Peak Value',
                hue='label', data=result_df, ax=ax, s=10)
plt.title('Average Entropy vs Peak Value')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.savefig('ent-peak-results.png')
plt.close()
