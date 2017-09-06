# Brief:     Test the densenet for image classification
# Data:      24/Aug./2017
# E-mail:    huyixuanhyx@gmail.com
# License:   Apache 2.0
# By:        Yeephycho @ Hong Kong

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_data_path", "./tfrecord", "training data dir")
tf.app.flags.DEFINE_string("log_dir", "./log", " the log dir")

tf.app.flags.DEFINE_integer("TRAINING_SET_SIZE", 2512, "total image number of training set")
tf.app.flags.DEFINE_integer("TESTING_SET_SIZE", 908, "total image number of training set")

tf.app.flags.DEFINE_integer("BATCH_SIZE", 16, "batch size")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 224, "image width and height")

tf.app.flags.DEFINE_float("INIT_LEARNING_RATE", 0.005, "initial learning rate")
tf.app.flags.DEFINE_float("DECAY_RATE", 0.5, "learning rate decay rate")
tf.app.flags.DEFINE_integer("DECAY_STEPS", 2000, "learning rate decay step")

tf.app.flags.DEFINE_float("weights_decay", 0.0001, "weights decay serve as l2 regularizer")
