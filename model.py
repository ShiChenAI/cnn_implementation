from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
import config
import json

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', int(config.get_configs('global.conf', 'dataset', 'batch_size')), """每个batch样本总数""")

IMAGE_HEIGHT = int(config.get_configs('global.conf', 'dataset', 'resize_image_height')) 
IMAGE_WIDTH = int(config.get_configs('global.conf', 'dataset', 'resize_image_width')) 
NUM_CLASSES = int(config.get_configs('global.conf', 'dataset', 'num_class'))
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = int(config.get_configs('global.conf', 'train', 'train_data_count'))
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(config.get_configs('global.conf', 'eval', 'eval_data_count'))

# Hyper parameters
MOVING_AVERAGE_DECAY = float(config.get_configs('global.conf', 'model', 'moving_average_decay'))
NUM_EPOCHS_PER_DECAY = float(config.get_configs('global.conf', 'model', 'num_epochs_per_decay'))
LEARNING_RATE_DECAY_FACTOR = float(config.get_configs('global.conf', 'model', 'learning_rate_decay_factor'))
INITIAL_LEARNING_RATE = float(config.get_configs('global.conf', 'model', 'initial_average_decay'))

TOWER_NAME = config.get_configs('global.conf', 'model', 'tower_name')

def activation_summary(x):
    """Activation summary for tensorboard
  
    """


    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def variable_on_cpu(name, shape, initializer):
    """initializer for CPU-train.
    
    """
    
    
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Initialization variable with weight decay.
    
    """


    var = variable_on_cpu(name, shape,tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def inference(images):
    """Inference of model.

    Arguments:
        images: Images returned from distorted_inputs() or inputs().
    """


