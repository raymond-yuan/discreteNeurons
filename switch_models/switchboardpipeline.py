import tensorflow as tf
import numpy as np
import os
from tqdm import trange
import matplotlib.pyplot as plt


class SwitchBoardPipeline:
    def __init__(self):
        self.EPOCHS = 100
        self.ALPHA = 0.1
        self.baselines = None
        self.batch_size = 200
        self.init_pipeline()

    def init_pipeline(self):
        self.load_data()
        self.load_model()
        assert hasattr(self, 'model'), "Model not defined during initialization"
        assert hasattr(self, 'num_examples'), "num_examples not defined during initialization"
        assert hasattr(self, 'x_train'), "x_train not defined during initialization"
        assert hasattr(self, 'x_test'), "x_test not defined during initialization"
        assert hasattr(self, 'y_train'), "y_train not defined during initialization"
        assert hasattr(self, 'y_test'), "y_test not defined during initialization"


    def batch_gen(self, data, labels, batch_size=32):
        while True:
            perm_ind = np.random.permutation(self.num_examples)
            for i in range(0, self.num_examples, batch_size):
                X_batch = data[perm_ind[i:i + batch_size]].astype('float32')
                X_batch /= 255.0
                y_batch = labels[perm_ind[i:i + batch_size]]

                yield X_batch, y_batch, perm_ind[i:i + batch_size]

    def load_data(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

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
        # print(self.model.weights[0].shape)
        # print(self.model.weights[1])
        grads = self.model.compute_grads(reward, y_batch)
        # print(grads)
        # print('weight')

        # print(self.model.weights)
        # weights_before = self.model.get_weights()
        self.opt.apply_gradients(zip(grads, self.model.weights))
        # for idx, w in enumerate(weights_before):
        #     assert np.any(self.model.get_weights()[idx] != w)
        return np.mean(R, 0)

    def train(self, save=True, test=False):
        lr = 1e-1
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

        num_batches = int(np.ceil(self.num_examples / self.batch_size))
        best_acc = -float('inf')
        for ep in range(self.EPOCHS):
            ds = self.batch_gen(self.x_train, self.y_train, batch_size=self.batch_size)
            t = trange(num_batches)
            epoch_acc = 0
            epoch_r = 0
            for batch in t:
                x_batch, y_batch, idxs = next(ds)
                batch_r = self.train_batch(x_batch, y_batch, idxs, ep == 0 and batch == 0)
                epoch_r += batch_r
                epoch_acc += 0.5 * (batch_r + 1)
                t.set_description("Ep: {}. Accuracy: {:.3f}, Rewards: {:.3f}".format(ep, epoch_acc / (batch + 1),
                                                                    epoch_r / (batch + 1)))

            if epoch_acc > best_acc and save:
                print('saving: {:.3f}'.format(epoch_acc / num_batches))
                best_acc = epoch_acc

                self.model.save_weights('./save_models/{}.h5'.format(self.model.__name__))

            if test:
                print(f"Test accuracy: {np.mean(np.argmax(self.model(self.x_test).numpy(), -1) == np.argmax(self.y_test, -1))}")

        print(f"Final Test accuracy: {np.mean(np.argmax(self.model(self.x_test).numpy(), -1) == np.argmax(self.y_test, -1))}")

    def visualize(self):
        model_path = '/Users/raymondyuan/Documents/projects/discreteNeurons/save_models/15:48:46.513301|500.h5'
        assert os.path.exists(model_path), 'Path does not exist'
        self.model.load_weights(model_path)
        for i, img in enumerate(self.x_test):
            try:
                plt.imshow(img)
                cat_test = np.argmax(self.y_test[i], -1)
                cat_predict = np.argmax(self.model(np.expand_dims(img, 0)).numpy(), -1)
                plt.title(f'Actual: {cat_test}, Predicted{cat_predict}')
                plt.show()
            except KeyboardInterrupt:
                print('Exiting')
                break