#!/usr/bin/env python

from net import build_net
from load_data import readImg
from sys import argv
import tflearn
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

netPath = "./net/dni_reader.tfl"
net = build_net()
net.load(netPath, weights_only=True)

def classify(imgPath):
    image = readImg(imgPath)
    data = np.array([image]).reshape([1, 150, 200, 1])
    prediction = net.predict(data)[0]
    return np.argmax(prediction)

print classify(argv[1])
