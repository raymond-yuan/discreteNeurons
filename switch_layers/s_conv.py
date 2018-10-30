from tensorflow.keras import initializers
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

tfe = tf.contrib.eager

class SConv(layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 padding='SAME'):
        self.kernel_size = kernel_size
        self.padding_type = padding
        self.trainable = True
        super(SConv, self).__init__(filters=filters,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    activation='sigmoid')

    def build(self, input_shape):
        super(SConv, self).build(input_shape)

    def call(self, inputs):
        self.inputs = inputs
        self.probs = super(SConv, self).call(inputs)
        dist = tf.distributions.Bernoulli(probs=self.probs, dtype=tf.float32)
        self.out = dist.sample()
        return self.out

    def compute_grads(self, reward, y_batch):
        if not isinstance(self.inputs, np.ndarray):
            _, iw, ih, ic = self.inputs.numpy().shape
        else:
            _, iw, ih, ic = self.inputs.shape
        _, ow, oh, oc = self.out.numpy().shape

        bias_update = tf.transpose(self.out - self.probs)
        bias_update *= (reward)

        w_update = -tf.nn.conv2d_backprop_filter(input=self.inputs,
                                                filter_sizes=self.kernel_size + (ic, oc),
                                                out_backprop=tf.transpose(bias_update),
                                                strides=(1, 1, 1, 1),
                                                padding=self.padding_type)
        # print("weight update", w_update.shape)
        # print("bias update shape:", tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(bias_update, -1), -1), -1).shape)
        #
        # print("input shape", self.inputs.shape)
        #
        # print([w.shape for w in self.weights])
        # print([w.name for w in self.weights])

        reduce_bias = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(bias_update, -1), -1), -1)
        return w_update, tfe.Variable(-reduce_bias)

