import tensorflow as tf
from PIL import Image
import os
import json
import sys
import argparse
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--num_index', type=int, default=5,
                    help='Number of records')
FLAGS = parser.parse_args()
DIR = "../data/train"
LabelStartIndexFile = "../data/train_label_v2.txt"

CLASS_NUMS = 128 #分类数
IMAGE_SIZE = 299 #图片大小
PER_CLASS_IMAGE_NUMS = 128
#说明：
# 将图片转换为records，每次每个分类提取100张图片，提取图片的起始位置由start_index指定，初始为0
#files_path 传入文件所在路径
#records_name records的名称 格式:"Kaggle_"+dir_name+"_"+start_index+".tfrecords"
#steps 训练次数
def convert_to_records(train_num):
    dir_path = DIR
    records_name = "../records/Kaggle_Train_"+str(train_num)+".tfrecords"
    writer = tf.python_io.TFRecordWriter(records_name)
    #读取train_label 文件，获取每种类型的起始img id
    fd = open(LabelStartIndexFile, "r")
    label_start_map = json.load(fd)

    for i in range(CLASS_NUMS):
        label = i#0-127
        for j in range(PER_CLASS_IMAGE_NUMS):
            class_num = str(label+1)
            start_img_id = label_start_map[class_num]["start"]
            end_img_id = label_start_map[class_num]["end"]
            img_id = int(start_img_id) + j + (train_num + 1) * PER_CLASS_IMAGE_NUMS

            if img_id > int(end_img_id):
                img_id = int(start_img_id) + (img_id - int(end_img_id))% (int(end_img_id) - int(start_img_id))

            #组装img name
            img_name = str(img_id)+"_"+class_num+".jpg"
            img_path = os.path.join(dir_path, img_name)
            #判断文件是否存在
            if os.path.exists(img_path):
                print(img_path)
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

convert_to_records(FLAGS.num_index)
