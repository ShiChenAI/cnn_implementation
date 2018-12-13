from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import math
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import model
import tfrecord
import config
import eval_model


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 
                           config.get_configs('global.conf', 'model', 'model_dir'),
                           """Train model checkpoint dir.""")

tf.app.flags.DEFINE_string('eval_data_dir', 
                           config.get_configs('global.conf', 'eval', 'eval_tfrecord_dir'),
                           """Evaluation data log.""")

tf.app.flags.DEFINE_integer('max_steps', 
                            int(config.get_configs('global.conf', 'train', 'max_steps')),
                            """Max training steps.""")

tf.app.flags.DEFINE_boolean('log_device_placement', 
                            False,
                            """Log device placement.""")

tf.app.flags.DEFINE_string('train_data',
                           config.get_configs('global.conf', 'train', 'train_tfrecord_dir'),
                           """Train data dir.""")

tf.app.flags.DEFINE_integer('train_num',
                            int(config.get_configs('global.conf', 'train', 'train_data_count')),
                           """Total number of train data.""")

tf.app.flags.DEFINE_integer('eval_num', 
                            int(config.get_configs('global.conf', 'eval', 'eval_data_count')),
                            """Total number of train data.""")
                
tf.app.flags.DEFINE_float('keep_prob',
                          float(config.get_configs('global.conf', 'model', 'keep_prob')),
                          """Keep probability in dropout computing.""")

tf.app.flags.DEFINE_boolean('create_train_eval_data', 
                            False,
                            """Create train data (80%) and evaluation data(20%) from original data.""")

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        # Get train images and labels
        float_image, label = tfrecord.train_data_read(tfrecord_path=FLAGS.train_data)
        images, labels = tfrecord.create_batch(float_image,label, count_num=FLAGS.train_num)

        # Get evaluate images and labels
        eval_float_image, eval_label = tfrecord.eval_data_read(tfrecord_path=FLAGS.eval_data_dir)
        eval_images, eval_labels = tfrecord.create_batch(eval_float_image, eval_label, count_num=FLAGS.eval_num)
        
        # Model inference
        keep_prob = tf.placeholder(tf.float32)
        #logits = model.inference(images, FLAGS.keep_prob)
        logits = model.inference(images, keep_prob)
        
        # loss computing
        loss = model.loss(logits, labels)

        # accuracy compution
        #accuracy = model.accuracy(model.inference(eval_images, 1), eval_labels)
        accuracy = model.accuracy(logits, eval_labels)

        # train model
        train_op = model.train(loss, global_step)

        # save model
        saver = tf.train.Saver(tf.global_variables())

        # merge all summaries
        summary_op = tf.summary.merge_all()
        
        # initialize all variables
        init = tf.initialize_all_variables()
        
        # Run session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # start queue runners
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                               graph_def=sess.graph_def)
        for step in xrange(FLAGS.max_steps):
            if step % 10 == 0:
                summary_str, acc = sess.run([summary_op, accuracy], feed_dict = {keep_prob: 1.0})
                summary_writer.add_summary(summary_str, step)
                print('%s: step %d, accuracy = %.3f' % (datetime.now(), step, acc))
            else:
                if step % 100 == 99 or (step + 1) == FLAGS.max_steps: 
                    summary_str, _ = sess.run([summary_op, train_op], feed_dict = {keep_prob: FLAGS.keep_prob})
                    summary_writer.add_summary(summary_str, step)
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                else:
                    start_time = time.time()
                    _, loss_value = sess.run([train_op, loss], feed_dict = {keep_prob: FLAGS.keep_prob})
                    duration = time.time() - start_time
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
                    

def main(argv=None): 
    if FLAGS.create_train_eval_data:
        tfrecord.create_train_eval_data()
        
    train()


if __name__ == '__main__':
    tf.app.run()