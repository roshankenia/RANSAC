import sys
sys.path.append('../')
from scipy.stats import entropy
import os
from tensorflow.keras.datasets import cifar10
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from cifar10_ransac_utils import *
from tensorflow.keras import losses
from ResNet import ResNet20ForCIFAR10
from tensorflow.keras.callbacks import LearningRateScheduler
import random



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


def makeConfidentTrainingSets(model, corTrainX, corTrainY, entropyThreshold):
    newTrainX = []
    newTrainY = []
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

        # if confident add to list
        if predictedClass == np.argmax(corTrainY[i]) and sampleEntropy <= entropyThreshold:
            newTrainX.append(corTrainX[i])
            newTrainY.append(corTrainY[i])

    return np.array(newTrainX), np.array(newTrainY)


# get data
cifar10_data = CIFAR10Data()
trainX, trainY, testX, testY = cifar10_data.get_data(subtract_mean=True)


# corrupt training data
noisePercentage = 0.25
corruptedTrainY = corruptData(trainY, noisePercentage)

# split data into training and validation
splitPercentage = 0.7
corTrainX, corTrainY, corValX, corValY, trainIndexes, valIndexes = splitTrainingData(
    trainX, corruptedTrainY, splitPercentage)


# compile a model
weight_decay = 1e-4
lr = 1e-1
num_classes = 10
corModel = ResNet20ForCIFAR10(input_shape=(
    32, 32, 3), classes=num_classes, weight_decay=weight_decay)
opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
corModel.compile(optimizer=opt,
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
r = corModel.fit(corTrainX, corTrainY, epochs=50,
                 batch_size=128, callbacks=[reduce_lr])


# obtain confident samples
entropyThreshold = 0.5
confTrainX, confTrainY = makeConfidentTrainingSets(
    corModel, corTrainX, corTrainY, entropyThreshold)

# compile a new model
weight_decay = 1e-4
lr = 1e-1
num_classes = 10
confModel = ResNet20ForCIFAR10(input_shape=(
    32, 32, 3), classes=num_classes, weight_decay=weight_decay)
opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
confModel.compile(optimizer=opt,
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])

# fit model to conf samples
r = confModel.fit(confTrainX, confTrainY, epochs=50,
                  batch_size=128, callbacks=[reduce_lr])

# obtain results
valAccuracy = confModel.evaluate(corValX, corValY)[1]

print('The trained model has an accuracy of',
      valAccuracy, 'on the testing data.')

# #save model
# cleanModel.save('ransac_clean.h5')
