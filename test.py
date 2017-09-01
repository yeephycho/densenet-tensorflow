# Brief:     Test the densenet for image classification
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
TEST_SET_SIZE = FLAGS.TESTING_SET_SIZE
BATCH_SIZE = FLAGS.BATCH_SIZE



def densenet_test():
    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    label_batch_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    if_training_placeholder = tf.placeholder(tf.bool, shape=[])

    image_batch, label_batch, filename_batch = data_provider.feed_data(if_random = False, if_training = False)
    label_batch_dense = tf.arg_max(label_batch, dimension = 1)

    if_training = tf.Variable(False, name='if_training', trainable=False)

    logits = tf.reshape(densenet.densenet_inference(image_batch_placeholder, if_training_placeholder, 1.0), [BATCH_SIZE, 5])
    logits_batch = tf.to_int64(tf.arg_max(logits, dimension = 1))

    correct_prediction = tf.equal(logits_batch, label_batch_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    checkpoint = tf.train.get_checkpoint_state("./models")
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        accuracy_accu = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        for i in range(int(TEST_SET_SIZE / BATCH_SIZE)):
            image_out, label_batch_dense_out, filename_out = sess.run([image_batch, label_batch_dense, filename_batch])
            print("label: ", label_batch_dense_out)
            accuracy_out, infer_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out,
                                                                                    label_batch_placeholder: label_batch_dense_out,
                                                                                    if_training_placeholder: if_training})
            accuracy_out = np.asarray(accuracy_out)
            print("infer: ", infer_out)
            print(' ')
            accuracy_accu = accuracy_out + accuracy_accu

        print(accuracy_accu / TEST_SET_SIZE * BATCH_SIZE)

        tf.train.write_graph(sess.graph_def, 'graph/', 'my_graph.pb', as_text=False)

        coord.request_stop()
        coord.join(threads)
        sess.close()
    return 0



def main():
    tf.reset_default_graph()

    densenet_test()



if __name__ == '__main__':
    main()
