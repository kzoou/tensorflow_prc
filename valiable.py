# coding: UTF-8

import tensorflow as tf

var1 = tf.Variable(0)
const2 = tf.constant(1)

# var1にconst2の値を足す
add_op = tf.add(var1, const2)
# 0 + 1 = 1
# 1 + 1 = 2

#assignメソッドは var1にadd_opを代入する処理。
update_var1 = tf.assign(var1, add_op)
# 0 ← 1を入れる。つまり1
# 1 ← 2を入れる。つまり2

mul_op = tf.multiply(add_op, update_var1)
# 1×1。つまり1
# 2×2。つまり4

init = tf.global_variables_initializer()

# 1つめのセッション
# セッション内で変数の状態を保持している。その為、実行毎に値が変わる。
with tf.Session() as sess:

    # 変数を初期化する
    sess.run(init)

    print(sess.run([mul_op]))
    print(sess.run([mul_op]))
    print(sess.run([mul_op]))
    print(sess.run([mul_op]))
    print(sess.run([mul_op]))

# 2つ目のセッション
# 1つ目のセッションとは別々に動作する。
with tf.Session() as sess:
    sess.run(init)
    print(sess.run([mul_op]))
    print(sess.run([mul_op]))