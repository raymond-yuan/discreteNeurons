import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import switch_models
from switch_models.switchboardpipeline import SwitchBoardPipeline


class SB_MLP(SwitchBoardPipeline):
    def __init__(self):
        super(SB_MLP, self).__init__()

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_test = self.x_test.astype(np.float32) / 255.0
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.num_examples = len(self.x_train)

    def load_model(self):
        dummy_data = tf.constant(np.random.random((10, 28, 28)), dtype=tf.float32)
        self.model = switch_models.SMLP()
        # Init model
        self.model.call(dummy_data)