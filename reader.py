# coding: UTF-8

import os
import numpy as np

# CIFAR-10形式のデータを読み込むやつ
# 読み込むデータセット（ファイル）を指定する →
# readメソッドに「レコード番号」を指定すると対応するレコード(Cifar10Record)が得られる

class Cifar10Record(object):
    width = 32
    height = 32
    depth = 3

    def set_label(self, label_byte):
        self.label = np.frombuffer(label_byte, dtype=np.uint8)

    def set_image(self, image_bytes):
        byte_buffer = np.frombuffer(image_bytes, dtype=np.int8)
        print(byte_buffer)
        reshaped_array = np.reshape(byte_buffer,
                                    [self.depth, self.height, self.width])
        self.byte_array = np.transpose(reshaped_array, [1, 2, 0])
        self.byte_array = self.byte_array.astype(np.float32)



class Cifar10Reader(object):

    def __init__(self, filename):
        if not os.path.exists(filename):
            print(filename + ' is not exist')
            return

        self.bytestream = open(filename, mode='rb')

    def close(self):
        if not self.bytestream:
            self.bytestream.close()

    def read(self, index):
        result = Cifar10Record()
        label_byte = 1
        image_bytes = result.height * result.width * result.depth
        record_bytes = label_byte + image_bytes

        self.bytestream.seek(record_bytes * index, 0)

        result.set_label(self.bytestream.read(label_byte))

        result.set_image(self.bytestream.read(image_bytes))

        return result