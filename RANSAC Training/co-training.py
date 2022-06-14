# -*- coding: utf-8 -*-
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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

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


def makeConfidentTrainingSets(firstConfidenceModel, secondConfidenceModel, corTrainX, corTrainY, entropyThreshold, peakThreshold, trainY):
    sampleArray = []
    classScores = [0, 0, 0, 0, 0]
    # find confident samples from first training set
    # obtain probability distribution of classes for each sample after the split and calculate its entropy
    # make predictions
    firstPredictions = firstConfidenceModel.predict(corTrainX)
    secondPredictions = secondConfidenceModel.predict(corTrainX)

    falseNegativeX = []
    falseNegativeY = []
    falseNegativeCount = 0

    # find entropy for every sample and decide if confident
    for i in range(len(corTrainX)):
        firstSample = firstPredictions[i]
        secondSample = secondPredictions[i]

        # get classification for each model
        firstPredictedClass = np.argmax(firstSample)
        secondPredictedClass = np.argmax(secondSample)

        # check if these two predictions are the same
        corresponding = False
        predictedClass = -1
        if firstPredictedClass == secondPredictedClass and firstPredictedClass == np.argmax(corTrainY[i]):
            corresponding = True
            predictedClass = firstPredictedClass

        # calculate average entropy
        sampleEntropy = (entropy(firstSample) + entropy(secondSample))/2

        # calculate average peak value
        firstProbSorted = sorted(firstSample)
        firstProbSorted = firstProbSorted[::-1]
        # sum all prob except max
        firstProbSum = 0
        for j in range(1, len(firstProbSorted)):
            firstProbSum += firstProbSorted[j]
        firstPeakValue = firstProbSorted[0]/firstProbSum
        if np.isnan(firstPeakValue) or firstPeakValue > 1000:
            firstPeakValue = 1000

        secondProbSorted = sorted(secondSample)
        secondProbSorted = secondProbSorted[::-1]
        # sum all prob except max
        secondProbSum = 0
        for j in range(1, len(secondProbSorted)):
            secondProbSum += secondProbSorted[j]
        secondPeakValue = secondProbSorted[0]/secondProbSum
        if np.isnan(secondPeakValue) or secondPeakValue > 1000:
            secondPeakValue = 1000

        peakValue = (firstPeakValue + secondPeakValue)/2

        confident = 0
        if corresponding and sampleEntropy <= entropyThreshold and peakValue >= peakThreshold:
            confident = 1

        # determine how accurate classification was

        classificationScore = 0
        if predictedClass == -1:
            classScores[4] += 1
        elif predictedClass != np.argmax(corTrainY[i]) and predictedClass != np.argmax(trainY[i]):
            classificationScore = 0
            classScores[0] += 1
        elif predictedClass == np.argmax(corTrainY[i]) and predictedClass != np.argmax(trainY[i]):
            classificationScore = 1
            classScores[1] += 1
        elif predictedClass != np.argmax(corTrainY[i]) and predictedClass == np.argmax(trainY[i]):
            classificationScore = 2
            classScores[2] += 1
        elif predictedClass == np.argmax(corTrainY[i]) and predictedClass == np.argmax(trainY[i]):
            classificationScore = 3
            classScores[3] += 1

        sampleData = [predictedClass, sampleEntropy,
                      peakValue, confident, classificationScore]
        sampleArray.append(sampleData)

        # if not confident but a clean label add to list (false negative)
        if confident == 0 and np.argmax(corTrainY[i]) == np.argmax(trainY[i]):
            falseNegativeX.append(corTrainX[i])
            falseNegativeY.append(corTrainY[i])
            falseNegativeCount += 1

    print('False negatives:', falseNegativeCount)
    print('Class Scores:', classScores)
    return sampleArray, falseNegativeX, falseNegativeY


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
addedInX = []
addedInY = []
for p in range(5):
    # select subset of data to train on
    # calculate number of samples to be added to subset
    numberTrain = int(0.5 * len(trainX))

    # generate indexes to use
    firstTrainIndexes = random.sample(
        range(0, len(trainX)), numberTrain)

    # add subset samples to correct arrays for first model
    firstSubsetTrainX = []
    firstSubsetTrainY = []
    for index in firstTrainIndexes:
        firstSubsetTrainX.append(trainX[index])
        firstSubsetTrainY.append(trainYMislabeled[index])
    # add in false negative samples to retrain on
    firstSubsetTrainX = firstSubsetTrainX + addedInX
    firstSubsetTrainY = firstSubsetTrainY + addedInY

    firstSubsetTrainX = np.array(firstSubsetTrainX)
    firstSubsetTrainY = np.array(firstSubsetTrainY)

    # get indexes for second model
    indexes = list(range(len(trainX)))
    setIndexes = set(indexes)
    setFirstTrainIndexes = set(firstTrainIndexes)
    secondTrainIndexes = list(setIndexes - setFirstTrainIndexes)
    # now get data for second model
    secondSubsetTrainX = []
    secondSubsetTrainY = []
    for index in secondTrainIndexes:
        secondSubsetTrainX.append(trainX[index])
        secondSubsetTrainY.append(trainYMislabeled[index])
    # add in false negative samples to retrain on
    secondSubsetTrainX = secondSubsetTrainX + addedInX
    secondSubsetTrainY = secondSubsetTrainY + addedInY

    secondSubsetTrainX = np.array(secondSubsetTrainX)
    secondSubsetTrainY = np.array(secondSubsetTrainY)

    # train model used to identify confident samples on first set of data
    firstConfidenceModel = trainModel(firstSubsetTrainX, firstSubsetTrainY)
    # train a model on second set of data
    secondConfidenceModel = trainModel(firstSubsetTrainX, firstSubsetTrainY)
    # from cross validation
    entropyThreshold = .1
    peakThreshold = 400

    # find samples that this model is confident on
    sampleArray, falseNegativeX, falseNegativeY = makeConfidentTrainingSets(
        firstConfidenceModel, secondConfidenceModel, trainX, trainYMislabeled, entropyThreshold, peakThreshold, trainY)

    # add iteration data to feature vector
    featureVector.append(sampleArray)

    # add false negative samples to add in list
    addedInX = addedInX + falseNegativeX
    addedInY = addedInY + falseNegativeY

# we first want to visualize the feature vector over the space

# lets first create a tensor for each sample over the iterations
# this vector will contain the stats for each sample
statVector = []
noiseVector = []
# iterate through each samples iteration data
confidentCount = 0
consistentCount = 0

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
    predLabels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for it in iterData:
        entVals.append(it[1])
        peakVals.append(it[2])
        predLabels[it[0]] += 1
        confidence += it[3]

        if it[0] != curLabel:
            consistent = 0
            consistentCount += 1

    # calculate avg entropy and peak
    avgEnt = np.average(entVals)
    avgPeak = np.average(peakVals)

    # calculate variance for entropy and peak
    stdEnt = np.std(entVals)
    stdPeak = np.std(peakVals)

    # give an ensemble label
    ensembleLabel = np.argmax(predLabels)

    # see if sample was confident in majority of predictions
    confident = 0
    if confidence > (len(featureVector)/2):
        confident = 1
        confidentCount += 1
        avgEnt = 0
        avgPeak = 0
        stdEnt = 0
        stdPeak = 0

    # add data to stat vector
    data = [avgEnt, avgPeak, stdEnt, stdPeak, confident]  # , consistent]
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
print('Number of samples that were inconsistent:', consistentCount)
print('Number of samples that were confident:', confidentCount)

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

# now lets run kmeans to see if we can distinguish the two groups
tsneData = pd.DataFrame(
    {'tSNE Feature 1': tsne_result[:, 0], 'tSNE Feature 2': tsne_result[:, 1]})
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(tsneData)
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='tSNE Feature 1', y='tSNE Feature 2',
                hue=kmeans.labels_, data=tsneData, ax=ax, s=10)
lim = (tsne_result.min()-5, tsne_result.max()+5)
plt.title('Predicted Label Using KMeans (2 Clusters)')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.savefig('kmeans-2-Results.png')
plt.close()
print('cluster center:', kmeans.cluster_centers_)
# calculate accuracy
normalCount = 0
inverseCount = 0
for l in range(len(kmeans.labels_)):
    label = kmeans.labels_[l]
    if label == noiseVector[l]:
        normalCount += 1
    else:
        if label == 0:
            if noiseVector[l] == 1:
                inverseCount += 1
        else:
            if noiseVector[l] == 0:
                inverseCount += 1
print('KMeans had a normal accuracy of:', normalCount, 'out of', len(
    noiseVector), 'which equals', (normalCount/len(noiseVector)))
print('KMeans had an inverse accuracy of:', inverseCount, 'out of', len(
    noiseVector), 'which equals', (inverseCount/len(noiseVector)))


# now we want to train on the clean data
# check which was more accurate
normAcc = (normalCount/len(noiseVector))
invAcc = (inverseCount/len(noiseVector))
inverse = False
if invAcc > normAcc:
    inverse = True

# create clean data arrays
cleanTrainX = []
cleanTrainY = []
for i in range(len(kmeans.labels_)):
    confLabel = kmeans.labels_[i]
    # if normal labelling add if confident
    if not inverse and confLabel == 1:
        cleanTrainX.append(trainX[i])
        cleanTrainY.append(trainYMislabeled[i])
    # if inverse labelling add if not 1
    elif inverse and confLabel == 0:
        cleanTrainX.append(trainX[i])
        cleanTrainY.append(trainYMislabeled[i])
cleanTrainX = np.array(cleanTrainX)
cleanTrainY = np.array(cleanTrainY)

# create and train a new model
cleanModel = trainModel(cleanTrainX, cleanTrainY)

# calculate accuracy of this model in using test data
accuracy = cleanModel.evaluate(testX, testY)[1]
print('This model had an accuracy of', accuracy, 'on the test data.')
