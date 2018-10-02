from tensorflow.keras import layers
from tensorflow.keras import models
import datetime

from smlp.s_dense import SDense


class SwitchBoardModel(models.Model):
    def __init__(self, size=500):
        super(SwitchBoardModel, self).__init__(name='sb')
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = SDense(size)
        self.dense2 = SDense(10, output_layer=True)
        self.__name__ = "{}|{}".format(str(datetime.datetime.now()).replace(' ', '|'), size)

    def call(self, inputs):
        flatten = self.flatten(inputs)
        x = self.dense1(flatten)
        x = self.dense2(x)
        return x


