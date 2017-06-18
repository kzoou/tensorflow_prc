# coding: UTF-8

import tensorflow as tf

# Tensorflowはデータフローグラフという形で計算を表現する
# 大規模な並列計算を行うのに有用とされている。

# 定数型のオペレーション（ノード）
const1 = tf.constant(2)
const2 = tf.constant(3)

# ノード同士を加算
add_op = tf.add(const1, const2)

# ノード同士を掛け算
mul_op = tf.multiply(add_op,const2)

# 複数のオペレーションを実行
with tf.Session() as sess:
    mul_result, add_result = sess.run([mul_op, add_op])

    print(mul_result)
    print(add_result)
