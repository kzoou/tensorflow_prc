# coding:UTF-8

# 推論（画像のクラス判別）をする
# どのクラスに属するのか推論するグラフを構築する。この推論グラフを「モデル」「ネットワーク」と呼ぶ
# 推論のモデルとしてCNNを使用する


import tensorflow as tf

# 正解ラベルの数かな？
NUM_CLASSES = 10


def _get_weights(shape, stddev=10):
    var = tf.get_variable(
        'weights',
        shape,

        initializer=tf.truncated_normal_initializer(stddev=stddev))

    return var


def _get_biases(shape, value=0.0):
    var = tf.get_variable(
        'biases',
        shape,
        initializer=tf.constant_initializer(value))

    return var

# inferenceが構築するグラフは、画像データ（image_node)を入力すると、
# 10個のfloat32型の要素を持つリスト（logits)を返す。
# logitsの値は、それぞれのクラスに対応しており、値が大きいほどそのクラスに近いと推論される
def inference(image_node):
    # 2つの畳み込み層、プーリング層、全結合層からなる。

    # conv1
    # 畳み込み層は画像から特徴量を抽出する
    with tf.variable_scope('conv1') as scope:
        # Rank4のTensor（多次元配列）5×5の大きさを持った、3チャンネルのフィルターを64枚
        # 重みを標準偏差0.1の正規分布で初期化
        weights = _get_weights(shape=[5, 5, 3, 64], stddev=1e-4)
        conv = tf.nn.conv2d(image_node,  weights, [1, 1, 1, 1],
                            padding='SAME')
        biases = _get_biases([64], value=0.1)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        wieghts = _get_weights(shape=[5, 5, 64, 64], stddev=1e-4)
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
        biases = _get_biases([64], value=0.1)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                       padding='SAME', name='pool2')

    reshape = tf.reshape(pool2, [1, -1])
    dim = reshape.get_shape()[1].value

    # fc3
    with tf.variable_scope('fc3') as scope:
        weights = _get_weights(shape=[dim, 384], stddev=0.04)
        biases = _get_biases([384], value=0.1)
        fc3 = tf.nn.relu(
            tf.matmul(reshape, weights) + biases,
            name=scope.name
        )

    # fc4
    with tf.variable_scope('fc4') as scope:
        weights = _get_weights(shape=[384, 192], stddev=0.04)
        biases = _get_biases([192], value=0.1)
        fc4 = tf.nn.relu(tf.matmul(fc3, weights) + biases, name=scope.name)

    # output
    with tf.variable_scope('output') as scope:
        weights = _get_weights(shape=[192, NUM_CLASSES], stddev=1 / 192.0)
        biases = _get_biases([NUM_CLASSES], value=0.0)
        logits = tf.add(tf.matmul(fc4, weights), biases, name='logits')

    return logits



