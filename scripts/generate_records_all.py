import tensorflow as tf
from PIL import Image
import os
import json
import sys
import argparse
parser = argparse.ArgumentParser()

# Basic model parameters.
FLAGS = parser.parse_args()
DIR = "../data/train"

CLASS_NUMS = 128 #分类数
IMAGE_SIZE = 299 #图片大小
PER_CLASS_IMAGE_NUMS = 128
#说明：
# 将图片转换为records，每次每个分类提取100张图片，提取图片的起始位置由start_index指定，初始为0
#files_path 传入文件所在路径
#records_name records的名称 格式:"Kaggle_"+dir_name+"_"+start_index+".tfrecords"
#steps 训练次数
def convert_to_records():
    dir_path = DIR
    records_name = "../records/Kaggle_Train_test.tfrecords"
    writer = tf.python_io.TFRecordWriter(records_name)
    list = os.listdir(dir_path)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        file_name = list[i]
        print(file_name)

        img_path = os.path.join(dir_path, file_name)
        if img_path.find(".jpg") != -1:
            print(img_path)
            label_name = file_name.split("_")
            img_id = label_name[0]
            label = int(label_name[1][:-4]) - 1
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
    writer.close()

def read_and_decode(filename_queue):
    """
    read and decode tfrecords
    """
    #    filename_queue = tf.train.string_input_producer([filename_queue])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'img_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    return image, label

convert_to_records()
