# coding: UTF-8

import tensorflow as tf

var1 = tf.Variable(0)

# 値が定まらない場合には型だけ指定したplaceholderを使う。具体的な値は実行時に与える。
holder2 = tf.placeholder(tf.int32)

add_op = tf.add(var1, holder2)
update_var1 = tf.assign(var1, add_op)

mul_op = tf.multiply(add_op, update_var1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # 実行時に値を与える
    result = sess.run(mul_op, feed_dict={
        holder2: 5
    })

    print(result)
    print(result)