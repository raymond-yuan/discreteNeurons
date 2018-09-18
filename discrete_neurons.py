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
        super(SDense, self).__init__(output_dim, kernel_initializer=init, activation=activation)

    def build(self, input_shape):
        super(SDense, self).build(input_shape)

    def call(self, inputs):
        self.inputs = inputs
        self.probs = super(SDense, self).call(inputs)
        bern = tf.distributions.Bernoulli(probs=self.probs)
        self.out = tf.to_float(bern.sample())
        return self.out

    def compute_grads(self, baseline, reward, predict, y_batch):

        if self.output_layer:
            x_t = tf.transpose(self.inputs)
            bias_update = y_batch - predict
            return tf.matmul(x_t, bias_update), tf.reduce_sum(bias_update, 0)
        else:
            x_t = tf.transpose(self.inputs) * (reward - baseline)
            bias_update = self.out - self.probs
            # print("xt:", x_t)
            # print()
            # print(bias_update)
            # print(tf.reduce_min(tf.matmul(x_t, bias_update)))
            # print(tf.reduce_max(tf.matmul(x_t, bias_update)))

            return tf.matmul(x_t, bias_update), tf.reduce_sum(bias_update, 0)


class SwitchBoard(models.Model):
    def __init__(self):
        super(SwitchBoard, self).__init__(name='sb')
        self.flatten = layers.Flatten(input_shape=(28, 28))
        self.dense1 = SDense(500)
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
    batch_size = 200
    opt = tf.train.GradientDescentOptimizer(1e1)
    lr = 1e1
    num_batches = int(np.ceil(num_examples / batch_size))

    for i in range(EPOCHS):
        ds = batch_gen(x_train, y_train, batch_size=batch_size)
        t = trange(num_batches)
        epoch_acc = 0
        for b in t:
            x_batch, y_batch = next(ds)
            predict = sb(x_batch).numpy()
            cat = np.argmax(predict, -1)
            cat_ybatch = np.argmax(y_batch, -1)
            reward = 2 * (cat == cat_ybatch) - 1
            epoch_acc += np.mean(cat == cat_ybatch)
            t.set_description("Accuracy: {:.5f}".format(epoch_acc / (b + 1)))
            # t.set_description("batch accuracy: {:.5f}".format(np.mean(cat == cat_ybatch)))
            if b == 0:
                baseline = -1
            else:
                baseline = (1 - ALPHA) * baseline + ALPHA * reward
            grad1 = sb.dense2.compute_grads(baseline, reward, predict, y_batch)
            grad2 = sb.dense1.compute_grads(baseline, reward, predict, y_batch)
            # print("grad2", tf.reduce_min(grad2[0]))
            # print("grad2", tf.reduce_max(grad2[0]))
            update1 = [w.numpy() - lr * g for w, g in zip(sb.dense2.weights, grad1)]
            update2 = [w.numpy() - lr * g for w, g in zip(sb.dense1.weights, grad2)]
            # sb.dense2.set_weights(update1)
            # sb.dense1.set_weights(update2)
            sb.set_weights(update2 + update1)
            # print(sb.weights)
                # v -= g * 1e-1
            # opt.apply_gradients(zip([*grad2, *grad1], sb.trainable_weights))
            # print(sb.weights[0])
            # opt.apply_gradients(zip(grad1, sb.dense2.weights))
            # print(weights_before)
            # print(sb.dense2.weights[0])
            # opt.apply_gradients(zip(grad2, sb.dense1.weights))
            # print('weights')
            # print(weights_before.numpy())
            # print()
            # print(sb.weights[2].numpy())
            # raise ValueError

        print(f"Test accuracy: {np.mean(np.argmax(sb(x_test).numpy(), -1) == np.argmax(y_test, -1))}")







