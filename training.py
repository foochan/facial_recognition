import random
import numpy as np
import cv2
import tensorflow as tf
from model import *
import flags
FLAGS = flags.Flags()

CHANNELS = FLAGS.CHANNELS
NUM_CLASSES = FLAGS.NUM_CLASSES
IMAGE_SIZE = FLAGS.IMAGE_SIZE
IMAGE_MATRIX_SIZE = IMAGE_SIZE*IMAGE_SIZE*CHANNELS
PATH_LABEL_FILE = "path_label.txt"

# データセットを作成する
PATH_AND_LABEL = []

# パスとラベルが記載されたファイルを一行ずつ読み込み、リストを作成する。
with open(PATH_LABEL_FILE, mode='r') as file:

    for line in file:
        # 改行文字を除く。
        line = line.rstrip()
        # スペースで区切り、配列にする。
        line_list = line.split()
        PATH_AND_LABEL.append(line_list)

# [画像情報, 正解ラベル]のペアを格納するためのリスト
DATA_SET = []

for path_label in PATH_AND_LABEL:
    tmp_list = []

    # 画像を読み込み、サイズを変更する。
    img = cv2.imread(path_label[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # 画像の情報を平坦化し、データ型も指定したうえで正規化
    img = img.flatten().astype(np.float32)/255.0
    tmp_list.append(img)

    # 分類するクラス数(NUM_CLASSES)の長さを持つ仮のリストを作成
    classes_array = np.zeros(NUM_CLASSES, dtype='float64')
    # 正解ラベルに該当する位置は1に上書き
    classes_array[int(path_label[1])] = 1
    tmp_list.append(classes_array)
    DATA_SET.append(tmp_list)

# データの分割
TRAIN_DATA_SIZE = int(len(DATA_SET)*0.8)
TRAIN_DATA_SET = DATA_SET[:TRAIN_DATA_SIZE]
TEST_DATA_SET = DATA_SET[TRAIN_DATA_SIZE:]

# バッチ数だけデータを取り出す関数
def batch_data(dataset, batchsize):
    dataset = random.sample(dataset, batchsize)

    return dataset

# 画像とラベルを分割する関数
def devide_dataset(dataset):
    dataset = np.array(dataset)   # ndarrayに変換

    image_dataset = dataset[:len(dataset), :1].flatten()
    label_dataset = dataset[:len(dataset), 1:].flatten()

    image_ndarray = np.empty((0, IMAGE_MATRIX_SIZE), dtype='float32')
    label_ndarray = np.empty((0, NUM_CLASSES))

    for (img, label) in zip(image_dataset, label_dataset):
        image_ndarray = np.append(image_ndarray, np.reshape(img, (1, IMAGE_MATRIX_SIZE)), axis=0)
        label_ndarray = np.append(label_ndarray, np.reshape(label, (1, NUM_CLASSES)), axis=0)

    return image_ndarray, label_ndarray

MAX_EPOCH = 1000
BATCH_SIZE = 3

############################ここからmain()######################################

x = tf.placeholder(tf.float32, [None, IMAGE_MATRIX_SIZE])   # 画像
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])   # ラベル
keep_prob = tf.placeholder("float")   # dropout率
y_conv = inference(x, keep_prob)   # 画像の畳み込み結果

# 交差エントロピー誤差
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

# 損失の最小化
with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))   # 結果と予測が一致するか否か
    correct_prediction = tf.cast(correct_prediction, tf.float32)   # bool値を0, 1に変換
accuracy = tf.reduce_mean(correct_prediction)    # 正解率
saver = tf.train.Saver()

# 訓練を実行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_step in range(MAX_EPOCH):

        train_dataset = batch_data(TRAIN_DATA_SET, BATCH_SIZE)
        train_image, train_label = devide_dataset(train_dataset)

        if epoch_step % BATCH_SIZE == 0:
            train_accuracy = accuracy.eval(feed_dict={x:train_image, y_:train_label, keep_prob:1.0})

        train_step.run(feed_dict={x:train_image, y_:train_label, keep_prob:0.5})

    test_image, test_label = devide_dataset(TEST_DATA_SET)

    saver.save(sess, "./model.ckpt")
