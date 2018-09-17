import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras import models
from  tensorflow.keras import initializers
import numpy as np
from tqdm import tqdm

tfe = tf.contrib.eager
tf.enable_eager_execution()

EPOCHS = 1000
ALPHA = 0.1

def logit(p):
    return np.log(p) - np.log(1 - p)


class SDense(layers.Layer):
    def __init__(self, output_dim, output_layer=False):
        super(SDense, self).__init__(name='sb_dense')
        self.probs = None
        if output_layer:
            init = initializers.glorot_uniform()
            activation = 'softmax'
        else:
            init = initializers.glorot_uniform()# initializers.RandomNormal(mean=logit(0.7))
            activation = 'sigmoid'
        self.dense = layers.Dense(output_dim, kernel_initializer=init, activation=activation)

    def call(self, inputs):
        self.inputs = inputs
        self.probs = self.dense(inputs)
        bern = tf.distributions.Bernoulli(probs=self.probs)
        self.out = bern.sample(sample_shape=self.probs.shape)
        return self.out

    def compute_grads(self, baseline, reward):
        x_t = tf.expand_dims(self.inputs, -1)
        return (reward - baseline) * (tf.matmul(x_t, tf.expand_dims(self.out)) - tf.matmul(x_t, self.probs))


class SwitchBoard(models.Model):
    def __init__(self):
        super(SwitchBoard, self).__init__(name='sb')
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = SDense(500)
        self.dense2 = SDense(10)

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
    num_examples = len(x_train)
    dummy_data = tf.constant(np.random.random((10, 28, 28)))
    sb = SwitchBoard()
    # Init model
    sb.call(dummy_data)
    batch_size = 128
    opt = tf.train.GradientDescentOptimizer(1e-1)
    for i in range(EPOCHS):
        ds = batch_gen(x_train, y_train, batch_size=batch_size)
        num_batches = int(np.ceil(num_examples / batch_size))
        for b in tqdm(range(num_batches)):
            x_batch, y_batch = next(ds)
            predict = np.argmax(sb(x_batch).numpy(), -1)
            reward = np.sum(2 * (predict == y_batch) - 1)
            print(f"accuracy: {np.mean(predict == y_batch)}")
            if b == 0:
                baseline = 0
            else:
                baseline = (1 - ALPHA) * baseline + ALPHA * reward
            grad1 = sb.dense2.compute_grads(baseline, reward)
            grad2 = sb.dense1.compute_grads(baseline, reward)
            opt.apply_gradients(zip(grad1, sb.dense2.weights))
            opt.apply_gradients(zip(grad2, sb.dense1.weights))







