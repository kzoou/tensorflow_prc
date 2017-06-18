# coding: UTF-8

import tensorflow as tf

# Tensorflowはデータフローグラフという形で計算を表現する
# 大規模な並列計算を行うのに有用とされている。

# 定数型のオペレーション（ノード）
const1 = tf.constant(2)
const2 = tf.constant(3)

# ノード同士を加算
add_op = tf.add(const1, const2)

# オペレーション（ノード）そのものを出力。値は出力されない。
print(add_op)


# オペレーションの演算結果を得るにはSessionオブジェクトを作る。
with tf.Session() as sess:

    # セッションオブジェクトを作ったらrunメソッドで計算。
    result = sess.run(add_op)
    print(result)


# pythonでは前処理や後処理を必要とするオブジェクトを扱う時に、それを簡易的に記述できる。
# それがwith文である。sess.close()とか書かなくていい。