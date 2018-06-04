# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the KAGGLE-128 binary file format."""

import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
# Process images of this size. Note that this differs from the original KAGGLE
# image size of 299 x 299. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 299

# Global constants describing the KAGGLE-128 data set.
NUM_CLASSES = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 200000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 6400
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 12647
def read_kaggle128(filename_queue):
    class KAGGLE128Record(object):
        pass
    result = KAGGLE128Record()

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'img_raw': tf.FixedLenFeature([], tf.string), 'label':tf.FixedLenFeature([], tf.int64)})
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    result.uint8image = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    result.label = tf.cast(features['label'], tf.int32)

    return result


IMAGE_SUMMARY_ADDED = False

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples,
            name='image_batch')
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            name='image_batch')

    # Display the training images in the visualizer.
    # global IMAGE_SUMMARY_ADDED
    # if not IMAGE_SUMMARY_ADDED:
    #   tf.image_summary('images', images)
        # IMAGE_SUMMARY_ADDED = True

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, train_start_num, batch_size):
    """Construct distorted input for KAGGLE training using the Reader ops.
    Args:
      data_dir: Path to the KAGGLE-128 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if train_start_num != -1:
        part_name = str(train_start_num)
    else:
        part_name = 'all'
    filenames = [os.path.join(data_dir, 'Kaggle_Train_' + part_name + '.tfrecords')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_kaggle128(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    padded_image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image, height + 4, width + 4)

    distorted_image = tf.random_crop(padded_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=0.25)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=0.8)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d KAGGLE images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(distorted_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    """Construct input for KAGGLE evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the KAGGLE-128 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if eval_data == "eval":
        filenames = [os.path.join(data_dir, 'Kaggle_Val.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    elif eval_data == "test":
        filenames = [os.path.join(data_dir, 'kaggle_test.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    else:
        filenames = [os.path.join(data_dir, 'kaggle_train.tfrecords')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_kaggle128(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image, width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image =  tf.image.per_image_standardization(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.2
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d KAGGLE images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
