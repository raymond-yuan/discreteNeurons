import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import initializers
import numpy as np
from tqdm import trange

tfe = tf.contrib.eager
tf.enable_eager_execution()

EPOCHS = 1000
ALPHA = 0.1

def logit(p):
    return np.log(p) - np.log(1 - p)


class SDense(layers.Layer):
    def __init__(self, output_dim, output_layer=False):
        super(SDense, self).__init__(name='sb_dense')
        self.output_layer = output_layer
        self.probs = None
        if output_layer:
            init = initializers.glorot_uniform()
            activation = 'softmax'
        else:
            init = initializers.RandomNormal(mean=logit(0.7))
            activation = 'sigmoid'
        self.dense = layers.Dense(output_dim, activation=activation)

    def build(self, input_shape):
        self.dense.build(input_shape=input_shape)
        super(SDense, self).build(input_shape)

    def call(self, inputs):
        self.inputs = inputs
        self.probs = self.dense(inputs)
        bern = tf.distributions.Bernoulli(probs=self.probs)
        self.out = tf.to_float(bern.sample())
        return self.out

    def get_weights(self):
        return self.dense.weights

    def compute_grads(self, baseline, reward, predict, y_batch):
        x_t = tf.expand_dims(self.inputs, -1)
        return tf.reduce_sum((reward - baseline) * (tf.matmul(x_t, tf.expand_dims(self.out, 1)) -
                                                    tf.matmul(x_t, tf.expand_dims(self.probs, 1))), 0), \
               tf.reduce_sum(y_batch - predict, 0) if self.output_layer else tf.reduce_sum(1 - self.probs, 0)


class SwitchBoard(models.Model):
    def __init__(self):
        super(SwitchBoard, self).__init__(name='sb')
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = SDense(10)
        self.dense2 = SDense(10, output_layer=True)

    def call(self, inputs):
        flatten = self.flatten(inputs)
        x = self.dense1(flatten)
        x = self.dense2(x)
        return x


def batch_gen(data, labels, batch_size=32):
    num_examples = data.shape[0]
    while True:
        perm_ind = np.random.permutation(num_examples)
        for i in range(0, num_examples, batch_size):
            X_batch = data[perm_ind[i:i + batch_size]].astype('float32')
            X_batch /= 255.0
            y_batch = labels[perm_ind[i:i + batch_size]]

            yield X_batch, y_batch


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    num_examples = len(x_train)
    dummy_data = tf.constant(np.random.random((10, 28, 28)), dtype=tf.float32)
    sb = SwitchBoard()
    # Init model
    sb.call(dummy_data)
    batch_size = 128
    opt = tf.train.GradientDescentOptimizer(1e-1)
    num_batches = int(np.ceil(num_examples / batch_size))

    for i in range(EPOCHS):
        ds = batch_gen(x_train, y_train, batch_size=batch_size)
        t = trange(num_batches)
        epoch_acc = 0
        for b in t:
            x_batch, y_batch = next(ds)
            predict = sb(x_batch).numpy()
            reward = np.sum(2 * (predict == y_batch) - 1)
            epoch_acc += np.mean(predict == y_batch)
            t.set_description(f"Accuracy: {epoch_acc / (b + 1)}")
            if b == 0:
                baseline = -1
            else:
                baseline = (1 - ALPHA) * baseline + ALPHA * reward

            grad1 = sb.dense2.compute_grads(baseline, reward, predict, y_batch)
            grad2 = sb.dense1.compute_grads(baseline, reward, predict, y_batch)
            opt.apply_gradients(zip(grad1, sb.dense2.get_weights()))
            opt.apply_gradients(zip(grad2, sb.dense1.get_weights()))
        print(f"Test accuracy: {np.mean(sb(x_test).numpy() == y_test)}")







