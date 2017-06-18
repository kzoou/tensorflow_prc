# coding: UTF-8

import tensorflow as tf

# TensorBoardに出力

# 定数型のオペレーション（ノード）
const1 = tf.constant(2)
const2 = tf.constant(3)

# 足し算
add_op = tf.add(const1, const2)

# 掛け算
mul_op = tf.multiply(add_op, const2)

with tf.Session() as sess:
    # 2つ同時に実行
    mul_result, add_result = sess.run([mul_op, add_op])
    print(mul_result)
    print(add_result)

    # グラフを出力。代入形式にしないと実行されない。
    file_writer = tf.summary.FileWriter('./', sess.graph)