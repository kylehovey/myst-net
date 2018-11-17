import os
import numpy as np
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

def genY(digit):
    return [
        1 if k == digit else 0 for k in range(25)
    ]

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
    map(lambda i: [ genY(i) ] * numTrain, range(25))
)

validX = flatten(
    map(
        lambda (i, samples): samples[-numValid:],
        enumerate(data)
    )
)

validY = flatten(
    map(lambda i: [ genY(i) ] * numValid, range(25))
)

def getData():
    return (
        np.array(trainX).reshape([-1, 150, 200, 1]),
        np.array(trainY),
        np.array(validX).reshape([-1, 150, 200, 1]),
        np.array(validY)
    )
