from tensorflow.keras import initializers
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

tfe = tf.contrib.eager

def logit(p):
    return np.log(p) - np.log(1 - p)


class SDense(layers.Dense):
    def __init__(self, output_dim, output_layer=False):
        self.output_layer = output_layer
        self.probs = None
        self.trainable = True
        if output_layer:
            init = initializers.glorot_uniform()
            activation = 'softmax'
        else:
            init = initializers.RandomNormal(mean=logit(0.7))
            activation = 'sigmoid'
        super(SDense, self).__init__(output_dim,
                                     kernel_initializer=init,
                                     activation=activation,
                                     bias_initializer=initializers.RandomNormal(mean=-2))

    def build(self, input_shape):
        super(SDense, self).build(input_shape)

    def call(self, inputs):
        self.inputs = inputs
        self.probs = super(SDense, self).call(inputs)
        if self.output_layer:
            dist = tf.distributions.Multinomial(1.0, probs=self.probs)
        else:
            dist = tf.distributions.Bernoulli(probs=self.probs)
        self.out = tf.to_float(dist.sample())
        return self.out

    def compute_grads(self, baseline, reward, y_batch):
        batch_size = len(reward)
        if self.output_layer:
            x_t = tf.transpose(self.inputs)
            bias_update = y_batch - self.probs
            return tfe.Variable(tf.matmul(x_t, bias_update) / batch_size),\
                   tfe.Variable(tf.reduce_mean(bias_update, 0))
        else:
            bias_update = tf.transpose(self.out - self.probs)  # 500, 200
            bias_update *= (reward - baseline)

            return tfe.Variable(tf.transpose(tf.matmul(bias_update, self.inputs)) / batch_size), \
                   tfe.Variable(tf.reduce_mean(bias_update, -1))
