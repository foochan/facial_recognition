import tensorflow as tf
import math
import flags
FLAGS = flags.Flags()

CHANNELS = FLAGS.CHANNELS
NUM_CLASSES = FLAGS.NUM_CLASSES
IMAGE_SIZE = FLAGS.IMAGE_SIZE

# 重みづけ値
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)   # 標準偏差0.1の正規分布乱数
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み処理
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max pooling(各範囲で最大値を選択して圧縮)
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference(images, keep_prob):

    # リサイズ
    x_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.variable_scope("conv1") as scope:
        W_conv1 = weight_variable([5, 5, CHANNELS, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope("conv2") as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2  = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # 全結合層1
    with tf.variable_scope("fc1") as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2
    with tf.variable_scope("fc2") as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_conv = tf.nn.softmax(logits)

    return y_conv
