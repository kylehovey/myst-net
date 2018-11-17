import os
import cv2
from functools import reduce

dataPath = "./data"
numTrain = 300
numValid = 400 - numTrain

def flatten(arr):
    return reduce(
        (lambda acc, y: acc + y),
        arr
    )

def readImg(path):
    image = cv2.imread(path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

data = []
for i in range(25):
    data.append([])

    for inst in range(400):
        path = "{}/{}/{}.png".format(dataPath, i, inst)
        data[i].append(readImg(path))

trainX = flatten(
    map(
        lambda (i, samples): samples[:numTrain],
        enumerate(data)
    )
)

trainY = flatten(
    map(lambda i: [ i ] * numTrain, range(25))
)

validX = flatten(
    map(
        lambda (i, samples): samples[numValid:],
        enumerate(data)
    )
)

validY = flatten(
    map(lambda i: [ i ] * numValid, range(25))
)

def getData():
    return (trainX, trainY, validX, validY)
