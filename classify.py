#!/usr/bin/env python

from net import build_net
from load_data import readImg
from sys import argv
import tflearn
import tensorflow as tf
import numpy as np
import pyfiglet

tf.logging.set_verbosity(tf.logging.ERROR)

netPath = "./net/dni_reader.tfl"
print "Building Network..."
net = build_net()
print "Loading Network Weights..."
net.load(netPath, weights_only=True)

def classify(image):
    print "Reading Image"
    data = np.array([image]).reshape([1, 150, 200, 1])

    print "Classifying"
    prediction = net.predict(data)[0]

    return np.argmax(prediction)

if __name__ == "__main__":
    image = readImg(argv[1])
    classification = classify(image)

    print ""
    print pyfiglet.figlet_format(str(classification), font="colossal")
