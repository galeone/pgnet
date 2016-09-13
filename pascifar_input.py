#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Generate the input from the PASCIFAR dataset"""

import os
import tensorflow as tf
import image_processing

# The depth of the example
INPUT_DEPTH = 3
INPUT_SIDE = 32

# Global constants describing the PASCIFAR data set.
NUM_CLASSES = 17
NUM_EXAMPLES = 42000


def read_pascifar(pascifar_path, queue):
    """ Reads and parses files from the queue.
    Args:
        pascifar_path: a constant string tensor representing the path of the PASCIFAR dataset
        queue: A queue of strings in the format: file, label

    Returns:
        image_path: a tf.string tensor. The absolute path of the image in the dataset
        label: a int64 tensor with the label
    """

    # Reader for text lines
    reader = tf.TextLineReader(skip_header_lines=1)

    # read a record from the queue
    _, row = reader.read(queue)

    # file,width,height,label
    record_defaults = [[""], [0]]

    image_path, label = tf.decode_csv(row, record_defaults, field_delim=",")

    image_path = pascifar_path + tf.constant("/") + image_path
    label = tf.cast(label, tf.int64)
    return image_path, label


def test(pascifar_path,
         batch_size,
         input_side,
         csv_path=os.path.abspath(os.getcwd())):
    """Returns a batch of images from the test dataset.

    Args:
        pascifar_path: path of the test dataset
        batch_size: Number of images per batch.
        input_side: resize images to shape [input_side, input_side, 3]
        csv_path: path (into the test dataset usually) where to find the list of file to read.
                        specify the filename and the path here, eg:
                         ~/data/PASCAL_2012/test/VOCdevkit/VOC2012/ImageSets/Main/test.txt
    Returns:
        images: Images. 4D tensor of [batch_size, input_side, input_side, DEPTH] size.
        filenames: file names. [batch_size] tensor with the fileneme read. (without extension)
    """

    pascifar_path = tf.constant(
        os.path.abspath(os.path.expanduser(pascifar_path)).rstrip("/") + "/")

    # read every line in the file, only once
    queue = tf.train.string_input_producer(
        [csv_path], num_epochs=1, shuffle=False, name="pascifar_queue")

    image_path, label = read_pascifar(pascifar_path, queue)

    # read, resize, scale between [-1,1]
    image = image_processing.eval_image(
        image_path, input_side, image_type="png")

    # create a batch of images & filenames
    # (using a queue runner, that extracts image from the queue)
    images, labels = tf.train.batch(
        [image, label],
        batch_size,
        shapes=[[input_side, input_side, INPUT_DEPTH], []],
        num_threads=1,
        capacity=20000,
        enqueue_many=False,
        name="pascifar_inputs")
    return images, labels
