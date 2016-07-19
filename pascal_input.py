#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Generate the input from the pascal dataset"""

import os
import tensorflow as tf
import pgnet
import image_processing

# Global constants describing the cropped pascal data set.
NUM_CLASSES = 20
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 293
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 27157

# sum = 27450 = PASCAL trainval size


def read_cropped_pascal(cropped_dataset_path, queue):
    """ Reads and parses files from the queue.
    Args:
        cropped_dataset_path: a constant string tensor representing the path of the cropped dataset
        queue: A queue of strings in the format: file, widht, height, label

    Returns:
        image_path: a tf.string tensor. The absolute path of the image in the dataset
        label: a int64 tensor with the label
        widht: a int64 tensor with the widht
        height: a int64 tensor with the height
    """

    # Reader for text lines
    reader = tf.TextLineReader(skip_header_lines=True)

    # read a record from the queue
    _, row = reader.read(queue)

    # file,width,height,label
    record_defaults = [[""], [0], [0], [0]]

    image_path, width, height, label = tf.decode_csv(
        row, record_defaults, field_delim=",")

    image_path = cropped_dataset_path + tf.constant("/") + image_path
    label = tf.cast(label, tf.int64)
    width = tf.cast(width, tf.int64)
    height = tf.cast(height, tf.int64)
    return image_path, label, width, height


def _generate_image_and_label_batch(image,
                                    label,
                                    min_queue_examples,
                                    batch_size,
                                    task='train'):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH] of type.float32.
      label: 1-D Tensor of type int64
      min_queue_examples: int64, minimum number of samples to retain
        in the queue that provides of batches of examples. The higher the most random (! important)
    batch_size: Number of images per batch.
    task: 'train' or 'validation'. In both cases use a shuffle queue
    Returns:
    images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    assert task == 'train' or task == 'validation'

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8

    images, sparse_labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    # add a scope to the summary. If shuffle=True, we're training
    # else we're validating
    tf.image_summary(task + '/images', images)

    return images, sparse_labels


def train(cropped_dataset_path,
          batch_size,
          csv_path=os.path.abspath(os.getcwd())):
    """Returns a batch of images from the train dataset.
    Applies random distortion to the examples.

    Args:
        cropped_dataset_path: path of the cropped pascal dataset
        batch_size: Number of images per batch.
        csv_path: path of train.csv
    Returns:
        images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH size.
        labes: Labels. 1D tensor of [batch_size] size.
    """
    cropped_dataset_path = os.path.abspath(
        os.path.expanduser(cropped_dataset_path)).rstrip("/") + "/"
    csv_path = csv_path.rstrip("/") + "/"

    # Create a queue that produces the filenames (and other atrributes) to read
    queue = tf.train.string_input_producer([csv_path + "train.csv"])

    # Read examples from the queue
    image_path, label, widht, height = read_cropped_pascal(
        tf.constant(cropped_dataset_path), queue)

    # read the image and cast it to float32
    # apply random distortions and resize image to:
    # pgnet.INPUT_SIDE x pgnet.INPUT_SIDE x pgnet.INPUT_DEPTH
    distorted_image = image_processing.train_image(
        image_path, widht, height, pgnet.INPUT_SIDE, image_type="jpg")

    # Ensure that the random shuffling has good mixing properties.
    fraction_of_examples_in_queue = 0.8
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             fraction_of_examples_in_queue)

    print(
        'Filling queue with {} pascal cropped images before starting to train. '
        'This will take a few....'.format(min_queue_examples))
    return _generate_image_and_label_batch(
        distorted_image, label, min_queue_examples, batch_size, task='train')


def validation(cropped_dataset_path,
               batch_size,
               csv_path=os.path.abspath(os.getcwd())):
    """Returns a batch of images from the validation dataset

    Args:
        cropped_dataset_path: path of the cropped pascal dataset
        batch_size: Number of images per batch.
        csv_path: path of valdation.csv
    Returns:
        images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH size.
        labes: Labels. 1D tensor of [batch_size] size.
    """

    cropped_dataset_path = os.path.abspath(
        os.path.expanduser(cropped_dataset_path)).rstrip("/") + "/"
    csv_path = csv_path.rstrip("/") + "/"

    queue = tf.train.string_input_producer([csv_path + "validation.csv"])

    # Read examples from files in the filename queue.
    image_path, label, _, _ = read_cropped_pascal(
        tf.constant(cropped_dataset_path), queue)

    # read and resize image
    image = image_processing.eval_image(
        image_path, pgnet.INPUT_SIDE, image_type="jpg")

    # Ensure that the random shuffling has good mixing properties.
    fraction_of_examples_in_queue = 0.8
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             fraction_of_examples_in_queue)

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(
        image, label, min_queue_examples, batch_size, task='validation')


def test(test_dataset_path,
         batch_size,
         file_list_path=os.path.abspath(os.getcwd())):
    """Returns a batch of images from the test dataset.

    Args:
        test_dataset_path: path of the test dataset
        batch_size: Number of images per batch.
        file_list_path: path (into the test dataset usually) where to find the list of file to read.
                        specify the filename and the path here, eg:
                         ~/data/PASCAL_2012/test/VOCdevkit/VOC2012/ImageSets/Main/test.txt
    Returns:
        images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH] size.
        filenames: file names. [batch_size] tensor with the fileneme read. (without extension)
    """

    test_dataset_path = os.path.abspath(os.path.expanduser(
        test_dataset_path)).rstrip("/") + "/"

    # read every line in the file, only once
    queue = tf.train.string_input_producer(
        [file_list_path], num_epochs=1, shuffle=False)

    # Reader for text lines
    reader = tf.TextLineReader()

    # read a record from the queue
    _, filename = reader.read(queue)

    image_path = test_dataset_path + tf.constant(
        "/JPEGImages/") + filename + tf.constant(".jpg")

    image = image_processing.read_image_jpg(image_path)
    image = image_processing.resize_bl(image, pgnet.INPUT_SIDE)

    # create a batch of images & filenames
    # (using a queue runner, that extracts image from the queue)
    images, filenames = tf.train.batch(
        [image, filename],
        batch_size,
        shapes=[[pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, pgnet.INPUT_DEPTH], []],
        num_threads=1,
        capacity=20000,
        enqueue_many=False)
    return images, filenames
