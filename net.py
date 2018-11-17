from load_data import getData
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Found on
# http://mckinziebrandon.me/TensorflowNotebooks/2016/11/28/early-stop-solution.html
class StopEarlyCallback(tflearn.callbacks.Callback):
    def __init__(self, thresh):
        self.thresh = thresh
    def on_epoch_end(self, state):
        if state.acc_value > self.thresh:
            raise StopIteration
    def on_train_end(self, state):
        "Stopped with accuracy {}".format(state.acc_value)

def build_net():
    network = input_data(shape=[None,50,38,1])
    network = conv_2d(network, 10, 25, activation="relu")
    network = max_pool_2d(network, 50)
    network = conv_2d(network, 5, 5, activation="relu")
    network = max_pool_2d(network, 10)
    network = fully_connected(network, 100, activation="relu")
    network = dropout(network, 0.6)
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
        show_metric=True,
        n_epoch=200,
        callbacks=StopEarlyCallback(0.9)
    )

    net.save("./net/dni_reader.tfl")
