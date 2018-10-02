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
            # init = initializers.RandomNormal(mean=logit(0.7))
            init = initializers.glorot_uniform()

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
        # print("probs", self.probs)
        batch_size = inputs.shape[0]
        if self.output_layer:
            dist = tf.distributions.Multinomial(1.0, probs=self.probs)

            # print("softmax", self.out)
        else:
            dist = tf.distributions.Bernoulli(probs=self.probs, dtype=tf.float32)

        self.out = dist.sample()
        if self.out.dtype != tf.float32:
            self.out = tf.to_float(dist.sample())

            # print("sigmoid", self.out)
        # self.out = tf.to_float(dist.sample())
        return self.out

    def compute_grads(self, baseline, reward, y_batch):
        batch_size = len(reward)
        if self.output_layer:
            x_t = tf.transpose(self.inputs)
            # bias_update[np.arange(batch_size), y_batch] += 1
            bias_update = y_batch - self.probs
            return tfe.Variable(-tf.matmul(x_t, bias_update)),\
                   tfe.Variable(-tf.reduce_sum(bias_update, 0))
        else:
            bias_update = tf.transpose(self.out - self.probs)
            # bias_update *= (reward)
            bias_update *= (reward)
            weight_grad = tfe.Variable(-tf.transpose(tf.matmul(bias_update, self.inputs)))
            # print(tf.reduce_max(weight_grad))
            # print(tf.reduce_min(weight_grad))
            # print(tf.reduce_mean(weight_grad))

            return weight_grad, \
                   tfe.Variable(-tf.reduce_sum(bias_update, -1))
