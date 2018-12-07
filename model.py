from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
import config
import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', int(config.get_configs('global.conf','dataset','batch_size')),"""每个batch样本总数""")
IMAGE_HEIGHT = int(config.get_configs('global.conf', 'dataset', 'resize_image_height')) 
IMAGE_WIDTH = int(config.get_configs('global.conf', 'dataset', 'resize_image_width')) 
NUM_CLASSES = int(config.get_configs('global.conf','dataset','num_class'))
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = int(config.get_configs('global.conf','train','train_data_count'))
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(config.get_configs('global.conf','eval','eval_data_count'))
