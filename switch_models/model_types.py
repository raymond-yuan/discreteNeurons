import datetime
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.training.checkpointable.data_structures import _ListWrapper

import switch_layers

class SwitchModel:
    def compute_grads(self, reward, y_batch, autograd=False):
        # print('Computing gradiests')
        grads = []
        # print(self.layers[0])
        # print(type(self.layers[0]))
        if isinstance(self.layers[0], _ListWrapper):
            iter_layers = self.layers[0]
        else:
            iter_layers = self.layers
        for layer in iter_layers:
            if getattr(layer, "compute_grads", None):
                # print('computing grads')
                grads.extend(layer.compute_grads(reward, y_batch))

        return grads


class SCNN(SwitchModel, models.Model):
    def __init__(self,
                 c_sizes=[32, 32, 64],
                 d_sizes=[512],
                 dropout=0.25):
        super(SCNN, self).__init__()
        # self.conv1 = switch_layers.SConv(32)
        # self.mp1 =
        self.call_layers = []
        for s in c_sizes:
            self.call_layers.append(switch_layers.SConv(s))
            self.call_layers.append(layers.MaxPool2D(pool_size=(2, 2)))
        self.call_layers.append(layers.Flatten())
        for s in d_sizes:
            self.call_layers.append(switch_layers.SDense(s))
        self.call_layers.append(switch_layers.SDense(10, output_layer=True))
        self.__name__ = "SCNN|{}|{}".format(str(datetime.datetime.now()).replace(' ', '|'), c_sizes)

    def call(self, inputs):
        x = inputs
        for layer in self.call_layers:
            x = layer(x)
        return x


class SMLP(SwitchModel, models.Model):
    def __init__(self, size=500):
        super(SMLP, self).__init__(name='sb')
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = switch_layers.SDense(size)
        self.dense2 = switch_layers.SDense(10, output_layer=True)
        self.__name__ = "SMLP|{}|{}".format(str(datetime.datetime.now()).replace(' ', '|'), size)

    def call(self, inputs):
        flatten = self.flatten(inputs)
        x = self.dense1(flatten)
        x = self.dense2(x)
        return x


