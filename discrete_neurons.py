import argparse

import tensorflow as tf

from switch_models.sb_cnn import SB_CNN
from switch_models.sb_mlp import SB_MLP

tfe = tf.contrib.eager
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--test_while_training', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--no_save', action='store_false')

args = parser.parse_args()

if __name__ == '__main__':
    sb = SB_CNN()
    # sb = SB_MLP()
    if args.train:
        sb.train(test=args.test_while_training, save=args.no_save)
    elif args.visualize:
        sb.visualize()




