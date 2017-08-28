# Brief:     Train a densenet for image classification
# Data:      24/Aug./2017
# E-mail:    huyixuanhyx@gmail.com
# License:   Apache 2.0
# By:        Yeephycho @ Hong Kong

# Code still under construction


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import net.densenet as densenet



DATA_DIR = "./tfrecord"
TRAINING_SET_SIZE = 2512
global_step = TRAINING_SET_SIZE * 100
TEST_SET_SIZE = 908
BATCH_SIZE = 16
IMAGE_SIZE = 224



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# image object from tfrecord
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.int32)

def read_and_decode(filename_queue):
    with tf.name_scope('data_provider'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features = {
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/height": tf.FixedLenFeature([], tf.int64),
            "image/width": tf.FixedLenFeature([], tf.int64),
            "image/filename": tf.FixedLenFeature([], tf.string),
            "image/class/label": tf.FixedLenFeature([], tf.int64),})
        image_encoded = features["image/encoded"]
        image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
        image_object = _image_object()
        image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
        image_object.height = features["image/height"]
        image_object.width = features["image/width"]
        image_object.filename = features["image/filename"]
        image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object



def flower_input(if_random = True, if_training = True):
    with tf.name_scope('image_reader_and_preprocessor'):
        if(if_training):
            filenames = [os.path.join(DATA_DIR, "train.tfrecord")]
        else:
            filenames = [os.path.join(DATA_DIR, "eval.tfrecord")]

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError("Failed to find file: " + f)
        filename_queue = tf.train.string_input_producer(filenames)
        image_object = read_and_decode(filename_queue)
        if(if_training):
            image = tf.image.random_flip_left_right(image_object.image)
            #    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
            image = tf.image.per_image_standardization(image)
        else:
            image = tf.image.per_image_standardization(image_object.image)

        label = image_object.label
        filename = image_object.filename

        if(if_random):
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
            print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
            num_preprocess_threads = 2
            image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
                [image, label, filename],
                batch_size = BATCH_SIZE,
                num_threads = num_preprocess_threads,
                capacity = min_queue_examples + 3 * BATCH_SIZE,
                min_after_dequeue = min_queue_examples)
            image_batch = tf.reshape(image_batch, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
            label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
            label_batch = tf.one_hot(tf.add(label_batch, label_offset), depth=5, on_value=1.0, off_value=0.0)
        else:
            image_batch, label_batch, filename_batch = tf.train.batch(
                [image, label, filename],
                batch_size = BATCH_SIZE,
                num_threads = 1)
            image_batch = tf.reshape(image_batch, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
            label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
            label_batch = tf.one_hot(tf.add(label_batch, label_offset), depth=5, on_value=1.0, off_value=0.0)
    return image_batch, label_batch, filename_batch



def flower_train():
    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 5])

    image_batch, label_batch, filename_batch = flower_input(if_random = True, if_training = True)

    logits = densenet.flower_inference(image_batch_placeholder, dropout_prob=0.7)

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=logits))
    #loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits)
    # create a summary for training loss
    tf.summary.scalar('loss', loss)

    train_step = tf.train.GradientDescentOptimizer(0.0004).minimize(loss)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    # sess = tf.Session()

    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    sess.run(tf.global_variables_initializer())

    if(tf.gfile.Exists("./models/flower.ckpt.meta")):
        saver.restore(sess, "./models/flower.ckpt")
        print("restoring model!")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)

    check_points = int(TRAINING_SET_SIZE/BATCH_SIZE)
    for epoch in range(200):
        for check_point in range(check_points):
            image_batch_train, label_batch_train, filename_train = sess.run([image_batch, label_batch, filename_batch])

            _, training_loss, summary = sess.run([train_step, loss, summary_op],
                                                   feed_dict={image_batch_placeholder: image_batch_train,
                                                              label_batch_placeholder: label_batch_train})

            if(bool(check_point%50 == 0) & bool(check_point != 0)):
                print("batch: ", check_point + epoch * check_points)
                print("training loss: ", training_loss)
                summary_writer.add_summary(summary, check_point + epoch * check_points)

        saver.save(sess, "./models/flower.ckpt")

    coord.request_stop()
    coord.join(threads)
    sess.close()
    return 0



def main():
    tf.reset_default_graph()
    flower_train()



if __name__ == '__main__':
    main()
