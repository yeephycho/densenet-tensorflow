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
import data_provider.data_provider as data_provider



DATA_DIR = "./tfrecord"
TRAINING_SET_SIZE = 2512
global_step = TRAINING_SET_SIZE * 100
TEST_SET_SIZE = 908
BATCH_SIZE = 16
IMAGE_SIZE = 224
learning_rate = 0.0004



def densenet_train():
    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 5])
    if_training_placeholder = tf.placeholder(tf.bool, shape=[])

    if_training = tf.Variable(True, name='if_training')

    image_batch, label_batch, filename_batch = data_provider.feed_data(if_random = True, if_training = True)

    logits = densenet.densenet_inference(image_batch_placeholder, if_training_placeholder, dropout_prob=0.7)

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=logits))
    #loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits)

    # create a summary for training loss
    tf.summary.scalar('loss', loss)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

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
                                                              label_batch_placeholder: label_batch_train,
                                                              if_training_placeholder: if_training})

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
    densenet_train()



if __name__ == '__main__':
    main()
