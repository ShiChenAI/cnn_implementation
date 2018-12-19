import tensorflow as tf
import os
import sys
import time
import random
from PIL import Image
import config

IMAGE_HEIGHT = int(config.get_configs('global.conf', 'dataset', 'resize_image_height')) 
IMAGE_WIDTH = int(config.get_configs('global.conf', 'dataset', 'resize_image_width')) 
CHANNELS = int(config.get_configs('global.conf', 'dataset', 'channels'))  
ORIGIN_DATASET = config.get_configs('global.conf', 'dataset', 'origin_data_dir') 
TRAIN_DATASET = config.get_configs('global.conf', 'dataset', 'train_data_dir') 
EVAL_DATASET = config.get_configs('global.conf', 'dataset', 'eval_data_dir') 
BATCH_SIZE = int(config.get_configs('global.conf', 'dataset', 'batch_size'))


def create(dataset_dir, tfrecord_path, tfrecord_name='train_tfrecord', width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Creat tfrecord dataset

    Arguments:
        dataset_dir: String, original data dir
        tfrecord_name: String, output tfrecord name
        tfrecord_path: String, output tfrecord path
        width: Integer, resize image width
        height: Integer, resize image height
    """


    if not os.path.exists(dataset_dir):
        print('Error! Original dataset path: %s does not exist..\n' % dataset_dir)
        exit()

    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(os.path.dirname(tfrecord_path))

    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path, tfrecord_name))
    lables = os.listdir(dataset_dir)
    print('%d labels to be classified.\n'% len(lables))

    for index, label in enumerate(lables):
        print('\nProcessing label: %s' % label)
        start_time = time.time()
        filepath = os.path.join(dataset_dir,label)
        filesNames = os.listdir(filepath)
        for i,file in enumerate(filesNames):
            imgPath = os.path.join(filepath,file)
            img = Image.open(imgPath)
            img = img.resize((width,height))
            img = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>> Converting image %d/%d , %g s' % (
                i+1, len(filesNames), time.time() - start_time))

    writer.close()
    print('\nFinished writing data to tfrecord files.')

def read(tfrecord_path, width, height, channels):
    """Read and pre-process images from tfrecord

    Arguments:
        tfrecord_path: String, tfrecord file path
        width: Integer, image width
        height: Integer, image height
        channels: Integer, image channels

    Returns: 
        img: image binary sequence
        label: String, image label
    """


    files = os.listdir(tfrecord_path)
    filenames = [os.path.join(tfrecord_path,tfrecord_name) for tfrecord_name in files]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=True)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'label': tf.FixedLenFeature([],tf.int64),
                                                 'image': tf.FixedLenFeature([],tf.string)})
    img = features['image'] 
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [channels,width,height])
    img = tf.transpose(img, [1, 2, 0]) # shape of tf.nn.conv2d() input is: [batch, in_height, in_width, in_channels]
    label = tf.cast(features['label'], tf.int32)

    return img, label


def data_process(img):
    """Process image

    Arguments:
        img: 3-D tensor of shape 

    Returns: 
        processed_image: processed image with same shape as input
    """


    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)  
    img = tf.image.random_brightness(img, max_delta=63)  
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)  
    processed_image = tf.image.per_image_standardization(img)  
    
    return processed_image


def train_data_read(tfrecord_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS, process_image=False):
    '''Read train data from tfrecord
    
    Arguments:
        tfrecord_path: String, tfrecord file path
        width: Integer, image width
        height: Integer, image height
        channels: Integer, image channels
        process_image: Boolen, process image or not

    Returns: 
        float_image
        label
    '''


    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)
        create(dataset_dir=TRAIN_DATASET,tfrecord_path=tfrecord_path)

    img, label = read(tfrecord_path, width, height, channels)
    
    if process_image:
        float_image = data_process(img)
    else:
        float_image = tf.cast(img, tf.float32)
    
    return float_image, label


def eval_data_read(tfrecord_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, channels=CHANNELS):
    '''Read evaluation data from tfrecord
    
    Arguments:
        tfrecord_path: String, tfrecord file path
        width: Integer, image width
        height: Integer, image height
        channels: Integer, image channels

    Returns: 
        float_image
        label
    '''


    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)
        create(dataset_dir=EVAL_DATASET,tfrecord_path=tfrecord_path)

    img, label = read(tfrecord_path, width, height, channels)
    img = tf.cast(img, tf.float32)
    float_image = tf.image.per_image_standardization(img)

    return float_image, label


def create_batch(float_image, label, count_num, batch_size=BATCH_SIZE):
    '''Creates batches by randomly shuffling tensors(batch < min_after_dequeue < capacity)
    
    Arguments:
        float_image: 3-D tensor of shape, sample image
        label: String, lable of float image
        count_num: Integer, number of all data
        batch_size: Integer, the new batch size pulled from the queue.

    Returns: 
        images
        label_batch
    '''


    capacity = int(count_num * 0.6 + 3 * BATCH_SIZE)
    min_after_dequeue = int(count_num * 0.6)
    images, label_batch = tf.train.shuffle_batch([float_image,label], batch_size=batch_size,
                                                 capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=5)
    tf.summary.image('images', images)

    return images, label_batch


def create_train_eval_data():
    """Create train data (80%) and evaluation data(20%) from original data.

    """


    if tf.gfile.Exists(TRAIN_DATASET):
        tf.gfile.DeleteRecursively(TRAIN_DATASET)
        
    if tf.gfile.Exists(EVAL_DATASET):
        tf.gfile.DeleteRecursively(EVAL_DATASET)
        
    tf.gfile.MkDir(TRAIN_DATASET) 
    tf.gfile.MkDir(EVAL_DATASET) 

    unique_labels = []
    dirs = os.listdir(ORIGIN_DATASET)
    for cur_dir in dirs:
        m = os.path.join(ORIGIN_DATASET, cur_dir)
        if (os.path.isdir(m)):
            h = os.path.split(m)
            unique_labels.append(h[1])

    for label in unique_labels:
        img_file_path = '%s/%s/*' % (ORIGIN_DATASET, label)
        matching_files = tf.gfile.Glob(img_file_path)
        count = int(0.8 * len(matching_files))
        train_indexs = random.sample(range(0, len(matching_files)), count)

        for index in range(len(matching_files)):
            if index in train_indexs:
                if not tf.gfile.Exists('%s/%s/' % (TRAIN_DATASET, label)):
                    tf.gfile.MkDir('%s/%s/' % (TRAIN_DATASET, label)) 
                new_path = '%s/%s/%s' % (TRAIN_DATASET, label, os.path.basename(matching_files[index]))
                tf.gfile.Copy(matching_files[index], new_path, overwrite = False)
            else:
                if not tf.gfile.Exists('%s/%s/' % (EVAL_DATASET, label)):
                    tf.gfile.MkDir('%s/%s/' % (EVAL_DATASET, label)) 
                new_path = '%s/%s/%s' % (EVAL_DATASET, label, os.path.basename(matching_files[index]))
                tf.gfile.Copy(matching_files[index], new_path, overwrite = False) 
            
            print('Copyed file from %s to %s ' % (matching_files[index], new_path))

def get_file_paths(file_dir, file_type):
    file_paths=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == file_type:
                file_paths.append(os.path.join(root, file))
    return file_paths

def get_file_name(file_path):
    (filepath,tempfilename) = os.path.split(file_path)
    (filename,extension) = os.path.splitext(tempfilename)
    return filepath, filename

def png2jpg():
    file_paths = get_file_paths('eval', '.png')
    for file_path in file_paths:
        im = Image.open(file_path)
        im = im.convert('RGB')
        new_path, new_name = get_file_name(file_path)
        im.save(new_path + '/' + new_name + '.jpg')
        os.remove(file_path)

def main():
    png2jpg()

if __name__ == '__main__':
    main()
