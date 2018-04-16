import tensorflow as tf
from PIL import Image
import os
import json
import sys

DIR = "../data/validation"

CLASS_NUMS = 128 #分类数
IMAGE_SIZE = 299 #图片大小
def convert_to_records():
    dir_path = DIR
    records_name = "../records/Kaggle_Val.tfrecords"
    writer = tf.python_io.TFRecordWriter(records_name)
    #读取文件夹下所有文件
    for img_name in os.listdir(dir_path):
        label = int(img_name.split("_")[1][:-4]) - 1
        img_path = os.path.join(dir_path, img_name)
        #判断文件是否存在
        if os.path.exists(img_path):
            try:
                # 读取图片并存储
                img = Image.open(img_path)
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_raw = img.tobytes()  # 将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())  # 序列化为字符串
            except Exception as e:
                print(e)
        else:
            print(img_path+", not exist!")
    writer.close()

convert_to_records()
