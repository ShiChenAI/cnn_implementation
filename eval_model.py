from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import model
import tfrecord
import config


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', 
                           config.get_configs('global.conf', 'eval', 'eval_log_dir'),
                           """Evaluation log dir.""")
tf.app.flags.DEFINE_string('eval_data', 
                           config.get_configs('global.conf', 'eval', 'eval_tfrecord_dir'),
                           """Evaluation data log.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 
                           config.get_configs('global.conf', 'model', 'model_dir'),
                           """Checkpoint dir.""")
tf.app.flags.DEFINE_integer('eval_interval_secs',
                            1,
                            """Evaluation interval time(sec)""")
tf.app.flags.DEFINE_integer('num_examples', 
                            int(config.get_configs('global.conf', 'eval', 'eval_data_count')),
                            """Total number of train data.""")
tf.app.flags.DEFINE_boolean('run_once', 
                            True,
                            """Only run once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Evaluation model onece

    """


    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return 
        
        coord = tf.train.Coordinator()
        
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))    # Iteration number
            true_count = 0  # True predicted count
            total_sample_count = num_iter * FLAGS.batch_size # Total evaluated data number 
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op]) #e.g. return [true,false,true,false,false]
                true_count += np.sum(predictions)
                step += 1
            
            # Compute precision
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval model for a number of steps.
  
    """


    with tf.Graph().as_default():
        float_image, label = tfrecord.eval_data_read(tfrecord_path=FLAGS.eval_data)
        images, labels = tfrecord.create_batch(float_image, label, count_num=FLAGS.num_examples)
        logits = model.inference(images, 1)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                               graph_def=graph_def)
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()