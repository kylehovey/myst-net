from load_data import getData
from net import build_net
import tflearn

netPath = "./net/dni_reader.tfl"
net = build_net()
net.load(netPath, weights_only=True)
trainX, trainY, validX, validY = getData()

print net.evaluate(trainX, trainY)
print net.evaluate(validX, validY)
