import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
from tqdm import trange

from smlp.model import SwitchBoardModel

tfe = tf.contrib.eager
tf.enable_eager_execution()


class SwitchBoard:
    def __init__(self):
        self.EPOCHS = 1000
        self.ALPHA = 0.1
        self.baselines = None
        self.batch_size = 200
        self.init_pipeline()

    def init_pipeline(self):
        self.load_data()
        self.load_model()

    def batch_gen(self, data, labels, batch_size=32):
        while True:
            perm_ind = np.random.permutation(self.num_examples)
            for i in range(0, self.num_examples, batch_size):
                X_batch = data[perm_ind[i:i + batch_size]].astype('float32')
                X_batch /= 255.0
                y_batch = labels[perm_ind[i:i + batch_size]]

                yield X_batch, y_batch, perm_ind[i:i + batch_size]

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_test = self.x_test.astype(np.float32) / 255.0
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        self.num_examples = len(self.x_train)

    def load_model(self):
        dummy_data = tf.constant(np.random.random((10, 28, 28)), dtype=tf.float32)
        self.model = SwitchBoardModel()
        # Init model
        self.model.call(dummy_data)

    def train_batch(self, x_batch, y_batch, idxs, first=False):
        predict = self.model(x_batch).numpy()
        cat = np.argmax(predict, -1)
        cat_ybatch = np.argmax(y_batch, -1)
        R = 2 * (cat == cat_ybatch) - 1

        if first:
            self.baselines = np.ones(self.num_examples, dtype=np.float32) * -1
            reward = R - self.baselines[idxs]
        else:
            reward = R - self.baselines[idxs]
            self.baselines[idxs] = (1 - self.ALPHA) * self.baselines[idxs] + self.ALPHA * reward

        grads = []
        for layer in self.model.layers[1:]:
            grads.extend(layer.compute_grads(self.baselines[idxs], reward, y_batch))

        weights_before = self.model.get_weights()

        updates = []
        # grads = [*grad1, *grad2]
        # dummy_a = tf.ones((784, 500))
        # grads[0] = dummy_a
        self.opt.apply_gradients(zip(grads, self.model.weights))
        # for idx, w in enumerate(self.model.get_weights()):
        #     updates.append(w - lr * grads[idx])
        # self.model.set_weights(updates)
        for idx, w in enumerate(weights_before):
            assert np.any(self.model.get_weights()[idx] != w)
        return np.mean(R, 0), np.mean(cat == cat_ybatch)


    def train(self, test=False):
        lr = 1e-1
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

        num_batches = int(np.ceil(self.num_examples / self.batch_size))

        for ep in range(self.EPOCHS):
            ds = self.batch_gen(self.x_train, self.y_train, batch_size=self.batch_size)
            t = trange(num_batches)
            epoch_acc = 0
            epoch_r = 0
            cat_acc = 0
            for batch in t:
                x_batch, y_batch, idxs = next(ds)
                batch_r, batch_acc = self.train_batch(x_batch, y_batch, idxs, ep == 0 and batch == 0)
                cat_acc += batch_acc
                epoch_r += batch_r
                epoch_acc += 0.5 * (batch_r + 1)
                t.set_description("Accuracy: {:.3f}, Rewards: {:.3f}".format(epoch_acc / (batch + 1),
                                                                    epoch_r / (batch + 1)))

            if test:
                print(f"Test accuracy: {np.mean(np.argmax(self.model(self.x_test).numpy(), -1) == np.argmax(self.y_test, -1))}")

        print(f"Final Test accuracy: {np.mean(np.argmax(self.model(self.x_test).numpy(), -1) == np.argmax(self.y_test, -1))}")


if __name__ == '__main__':
    sb = SwitchBoard()
    sb.train()




