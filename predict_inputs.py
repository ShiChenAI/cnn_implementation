from PIL import Image
import tensorflow as tf
import model
import config


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 
                           config.get_configs('global.conf', 'model', 'model_dir'),
                           """Checkpoint dir.""")
tf.app.flags.DEFINE_string('input_img', 
                           './eval/bar/211-2-2.jpg',
                           """input image to be predicted.""")
tf.app.flags.DEFINE_integer('image_height', 
                            int(config.get_configs('global.conf', 'dataset', 'resize_image_height')),
                            """Resized image height.""")

tf.app.flags.DEFINE_integer('image_width', 
                            int(config.get_configs('global.conf', 'dataset', 'resize_image_width')),
                            """Resized image width.""")


def inputs(input, count=1, batch_size=1):
    """Get input image.

    """


    model.FLAGS.batch_size = batch_size
    img = Image.open(input)
    img = img.resize((FLAGS.image_width, FLAGS.image_height))
    img = img.tobytes()
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [3, FLAGS.image_width, FLAGS.image_height])
    img = tf.transpose(img, [1, 2, 0])
    img = tf.cast(img, tf.float32)
    float_image = tf.image.per_image_standardization(img)
    capacity = int(count * 0.4 + 3 * batch_size)
    min_after_dequeue = int(batch_size * 0.4)
    images, _ = tf.train.shuffle_batch([float_image, '?'], 
                                       batch_size=batch_size,
                                       capacity=capacity, 
                                       min_after_dequeue=min_after_dequeue, 
                                       num_threads=5)

    return images


def predict(imgPath):
    """Predict input image.

    """


    with tf.Graph().as_default():
        float_image = inputs(imgPath)
        logits = model.inference(float_image, 1)
        top_k_op = tf.nn.in_top_k(logits, [1], 1)
        variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter('./pred',
                                               graph_def=graph_def)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                predictions = sess.run([logits])  # e.g. return [true,false,true,false,false]
                print(predictions)
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=1)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):
    predict(FLAGS.input_img)


if __name__ == '__main__':
    tf.app.run()