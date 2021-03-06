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

"""Evaluation for KAGGLE-128."""
import csv
import os
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import kaggle128
from nets import inception

parser = kaggle128.parser

parser.add_argument('--eval_dir', type=str, default='/Users/luocf/workspace/Kaggle/records',
                    help='Directory where to write event logs.')

parser.add_argument('--eval_data', type=str, default='test',
                    help='Either `test` or `train_eval`.')

parser.add_argument('--checkpoint_dir', type=str, default='/Users/luocf/workspace/Kaggle/source/inception_v2/eval_checkpoint',
                    help='Directory where to read model checkpoints.')

parser.add_argument('--eval_interval_secs', type=int, default=60 * 5,
                    help='How often to run the eval.')

parser.add_argument('--num_examples', type=int, default=12800,
                    help='Number of examples to run.')

parser.add_argument('--run_once', type=bool, default=False,
                    help='Whether to run eval only once.')

def eval_once(saver, predictions, labels):
    """Run Eval once.
    Args:
      saver: Saver.
      predictions: Top K op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/kaggle128_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            step = 0
            img_id = 0
            outfile = os.path.join(FLAGS.eval_dir, "submission_randomlabel.csv")
            out = open(outfile, 'w', newline='')
            csv_writer = csv.writer(out,dialect='excel')
            csv_writer.writerow(['id','predicted'])
            while step < num_iter and not coord.should_stop():
                output = sess.run(predictions)
                print(labels)
                for label in output:
                    img_id = img_id+1
                    result = str(img_id)+','+str(label+1)
                    print(result)
                    csv_writer.writerow([str(img_id),str(label+1)])
                #true_count += np.sum(output)
                #print(predictions)
                step += 1

            # Compute precision @ 1.
            #precision = true_count / total_sample_count
            #print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval KAGGLE-128 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for KAGGLE-128.
        eval_data = FLAGS.eval_data
        images, labels = kaggle128.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, end_points = inception.inception_resnet_v2(images, num_classes=kaggle128.NUM_CLASSES,
                                                           create_aux_logits=False,
                                                           is_training=False)
        # Calculate predictions.
        predictions = tf.argmax(logits, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            kaggle128.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        eval_once(saver, predictions, labels)

def main(argv=None):  # pylint: disable=unused-argument
    evaluate()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
