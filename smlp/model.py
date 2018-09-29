from tensorflow.keras import layers
from tensorflow.keras import models

from smlp.s_dense import SDense


class SwitchBoardModel(models.Model):
    def __init__(self):
        super(SwitchBoardModel, self).__init__(name='sb')
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = SDense(500)
        self.dense2 = SDense(10, output_layer=True)

    def call(self, inputs):
        flatten = self.flatten(inputs)
        x = self.dense1(flatten)
        x = self.dense2(x)
        return x