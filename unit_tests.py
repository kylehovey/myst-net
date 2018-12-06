# This Python file uses the following encoding: utf-8

from __future__ import print_function
from contextlib import contextmanager
import numpy as np
import tflearn
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from load_data import readImg, getData
from net import build_net
from classify import classify_with_net

# <=< Printing Utility Functions >=>
# Color printing derived from:
# https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
@contextmanager
def printESC(prefix, color, text):
    print("{prefix}{color}{text}".format(prefix=prefix, color=color, text=text), end='')
    yield
    print("{prefix}0m".format(prefix=prefix))

def printColor(string, color):
    with printESC('\x1B[', color, string):
        pass

# <=< Testing Utility Functions >=>
class Test():
    def __init__(self):
        self.tests = []
    def it(self, description, name, callback):
        def run():
            success = callback()

            if success:
                printColor("It {}: 👍".format(description), "38;32m")
            else:
                printColor("It {}: 🌩".format(description), "38;31m")

        self.tests.append((run, name))
    def runTests(self):
        nTests = len(self.tests)
        print("{} tests to run".format(nTests))
        print("<==================>")
        for i, (test, name) in enumerate(self.tests):
            print("Running test {} of {}: {}".format(i + 1, nTests, name))
            test()
            print("-------------------")

# If you load the same network twice, you get an error. This function will load
# the network if it hasn't been loaded before, but if it has, return it
def memoizeNet(store = {"net": None}):
    if store["net"] is None:
        netPath = "./net/dni_reader.tfl"
        net = build_net()
        net.load(netPath, weights_only=True)

        store["net"] = net

    return store["net"]

def testLoadImage():
    path = "./data/23/130.png"
    img = readImg(path)

    return img.shape == (150, 200)

def testLoadData():
    trainX, trainY, validX, validY = getData()

    return (
        trainX.shape == (7500, 150, 200, 1) and
        trainY.shape == (7500, 25) and
        validX.shape == (2500, 150, 200, 1) and
        validY.shape == (2500, 25)
    )

def testLoadNet():
    try:
        net = memoizeNet()

        return True
    except:
        return False


def testRunNet():
    net = memoizeNet()

    path = "./data/23/130.png"
    img = readImg(path)
    classification = classify_with_net(net, img, verbose=False)

    return classification == 23

def testTestingAccuracy():
    net = memoizeNet()
    trainX, trainY, validX, validY = getData()

    [ accuracy ] = net.evaluate(trainX, trainY)
    printColor("Training Accuracy: {:.2f}%".format(accuracy * 100), "38;32m")

    return accuracy > 0.95

def testValidationAccuracy():
    net = memoizeNet()
    trainX, trainY, validX, validY = getData()

    [ accuracy ] = net.evaluate(validX, validY)
    printColor("Validation Accuracy: {:.2f}%".format(accuracy * 100), "38;32m")

    return accuracy > 0.95

if __name__ == "__main__":
    testing = Test()

    testing.it("loads images from the filesystem", "Image Loading", testLoadImage)
    testing.it("loads and formats all training and validation data", "Data Loading", testLoadData)
    testing.it("loads the network itself", "Network Loading", testLoadNet)
    testing.it("runs the network on input data", "Network Running", testRunNet)
    testing.it("is at least 95% accurate on testing data", "Testing Accuracy", testTestingAccuracy)
    testing.it("is at least 95% accurate on validation data", "validation Accuracy", testValidationAccuracy)

    testing.runTests()
