# Brief:     Train a densenet for image classification
# Data:      24/Aug./2017
# E-mail:    huyixuanhyx@gmail.com
# License:   Apache 2.0
# By:        Yeephycho @ Hong Kong

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import net.densenet as densenet
import net.config as config
import data_provider.data_provider as data_provider



FLAGS = tf.app.flags.FLAGS
TRAINING_SET_SIZE = FLAGS.TRAINING_SET_SIZE
BATCH_SIZE = FLAGS.BATCH_SIZE



def densenet_train():
    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 5])
    if_training_placeholder = tf.placeholder(tf.bool, shape=[])

    if_training = tf.Variable(True, name='if_training')
    # if_training = tf.constant(True, dtype=tf.bool, shape=None, name='if_training', verify_shape=False)

    image_batch, label_batch, filename_batch = data_provider.feed_data(if_random = True, if_training = True)

    logits = densenet.densenet_inference(image_batch_placeholder, if_training_placeholder, dropout_prob=0.7)

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=logits))
    #loss = tf.losses.mean_squared_error(labels=label_batch_placeholder, predictions=logits)
    tf.summary.scalar('loss', loss) # create a summary for training loss

    global_step = tf.Variable(0, trainable=False)
    current_step = tf.assign(global_step, global_step+1)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                               global_step=current_step,
                                               decay_steps=3000,
                                               decay_rate=0.6,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    summary_op = tf.summary.merge_all()  # merge all summaries into a single "operation" which we can execute in a session

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    # sess = tf.Session()

    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    sess.run(tf.global_variables_initializer())

    checkpoint = tf.train.get_checkpoint_state("./models")
    if(checkpoint != None):
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)

    check_points = int(TRAINING_SET_SIZE/BATCH_SIZE)
    for epoch in range(250):
        for check_point in range(check_points):
            image_batch_train, label_batch_train, filename_train = sess.run([image_batch, label_batch, filename_batch])

            _, training_loss, summary, _current_step = sess.run([train_step, loss, summary_op, current_step],
                                                 feed_dict={image_batch_placeholder: image_batch_train,
                                                            label_batch_placeholder: label_batch_train,
                                                            if_training_placeholder: if_training})

            if(bool(check_point%50 == 0) & bool(check_point != 0)):
                print("batch: ", check_point + epoch * check_points)
                print("training loss: ", training_loss)
                summary_writer.add_summary(summary, _current_step)

        saver.save(sess, "./models/densenet.ckpt", _current_step)

    coord.request_stop()
    coord.join(threads)
    sess.close()
    return 0



def main():
    tf.reset_default_graph()
    densenet_train()



if __name__ == '__main__':
    main()
