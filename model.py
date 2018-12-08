from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
import config

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
KEEP_PROP = float(config.get_configs('global.conf', 'model', 'keep_prop'))

TOWER_NAME = config.get_configs('global.conf', 'model', 'tower_name')

def activation_summary(x):
    """Activation summary for tensorboard
  
    """


    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def variable_with_weight_decay(name, shape, stddev, wd):
    """Initialization variable with weight decay.
    
    """


    var = tf.get_variable(name, shape,tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def bias_variable(shape, init_value):
    """Get bias variable

    """


    return tf.get_variable(name='biases', shape, tf.constant_initializer(init_value))
 

def conv2d(inputs, kernel, s, bias, name):
    """Computes a 2-D convolution on the 4-D input.

    Arguments:
        inputs: 4-D Tensor, input images.
        kernel: 4-D Tensor, convolutional kernel.
        s: Integer, stride of the sliding window.
        bias: 1-D Tensor, bias to be added.
        name: String, optional name for the operation.

    Returns:
        The convolutioned output tensor.
    """


    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, kernel, strides=[1, s, s, 1], padding='SAME'), bias), name=name)
 

def max_pool(l_input, k1, k2, name):
    """Performs the max pooling on the input.

    Arguments:
        l_input: 4-D Tensor, input layer.
        k1: Integer, size of the window.
        k2: Integer, stride of the sliding window.
        name: String, optional name for the operation.

    Returns:
        The max pooled output tensor.
    """


    return tf.nn.max_pool(l_input, ksize=[1, k1, k1, 1], strides=[1, k2, k2, 1], padding='SAME', name=name)


def inference(images, keep_prop):
    """Inference of model.

    Arguments:
        images: 4-D Tensor, input images.
        keep_prop: Float, keep probability in dropout computing.

    Returns:
        dense: Tensor, outputs.
    """


    # Get network parameters from configuration file
    layers, weights, biases = config.get_network('network.json')

    inputs = images
    for index, layer in enumerate(layers):
        if layer.startswith('conv'):
            # Conv layers
            with tf.variable_scope(layer) as scope:
                kernel = variable_with_weight_decay('weights', shape=weights['w'+layer], stddev=1e-4, wd=0.0)
                bias = bias_variable(biases['b'+layer])
                conv = conv2d(inputs, kernel, 1, bias, name=scope)
                activation_summary(conv)
            
            pool = max_pool(conv, 3, 2, name='pool'+str(index+1))
            norm = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm'+str(index+1))
            inputs = norm
        else:
            # FC layers
            with tf.variable_scope(layer) as scope:
                weight = variable_with_weight_decay('weights', shape=weights['w'+layer], stddev=0.04, wd=0.0004)
                bias = bias_variable(biases['b'+layer])
                if layer.endswith('1'):
                    reshape = tf.reshape(inputs, [-1, weight.get_shape().as_list()[0]])
                    dense =  tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, weight) + bias), keep_prob, name=scope) 
                else:
                    dense =  tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, weight) + bias), keep_prob, name=scope)

            inputs = dense

        return dense


def loss(logits, labels):
    """L2 error computing.

    """


    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, NUM_CLASSES],1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = dense_labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def add_loss_summaries(total_loss):
    """Add loss summaries for Tensorboard.
  
    """


    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    
    return loss_averages_op


def train(total_loss, global_step):
    """Train model.
  
    """


    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)
    loss_averages_op = add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op