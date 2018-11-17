from load_data import getData
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def build_net():
    network = input_data(shape=[None,150,200,1])
    network = conv_2d(network, 10, 10, activation="elu")
    network = max_pool_2d(network, 100)
    network = conv_2d(network, 5, 5, activation="elu")
    network = max_pool_2d(network, 10)
    network = fully_connected(network, 100, activation="elu")
    network = dropout(network, 0.5)
    network = fully_connected(network, 25, activation="softmax")
    network = regression(
        network,
        optimizer='adam',
        learning_rate=0.001,
        loss='categorical_crossentropy'
    )

    return tflearn.DNN(network)

if __name__ == "__main__":
    trainX, trainY, validX, validY = getData()

    print trainX.shape
    print trainY.shape
    print validX.shape
    print validY.shape

    net = build_net()
    net.fit(
        trainX,
        trainY,
        validation_set=(validX, validY),
        show_metric=True
    )

    net.save("./net/dni_reader.tfl")
