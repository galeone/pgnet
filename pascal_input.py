import os
import tensorflow as tf
import pgnet

# The depth of the example
DEPTH = 3

# Global constants describing the cropped pascale data set.
NUM_CLASSES = 20
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 23758
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 11918


def read_cropped_pascal(base_path, queue):
    """ Reads and parses files from the queue.
    Args:
        base_path: a constant string tensor representing the path of the cropped dataset
        queue: A queue of strings in the format: file, widht, height, label

    Returns:
        An object representing a single example, with the following fields:
        key: a scalar string Tensor describing the filename & record number for this example.
        label: a tensor int32 with the label
        image: a [pgnet.INPUT_SIDE; pgnet.INPUT_SIDE, DEPTH] float32 tensor with the image data, resized with nn interpolation
    """

    class PASCALCroppedRecord(object):
        pass

    result = PASCALCroppedRecord()

    # Reader for text lines
    reader = tf.TextLineReader(skip_header_lines=True)

    # read a record from the queue
    result.key, row = reader.read(queue)

    # file,width,height,label
    record_defaults = [[""], [0], [0], [0]]

    image_path, _, _, result.label = tf.decode_csv(
        row, record_defaults, field_delim=",")

    image_path = base_path + tf.constant("/") + image_path
    image = tf.image.decode_jpeg(tf.read_file(image_path))

    #reshape to a 4-d tensor (required to resize)
    image = tf.expand_dims(image, 0)

    # now image is 4-D float32 tensor: [1,pgnet.INPUT_SIDE,pgnet.INPUT_SIDE, DEPTH]
    image = tf.image.resize_nearest_neighbor(image, [pgnet.INPUT_SIDE,
                                                     pgnet.INPUT_SIDE])
    # remove the 1st dimension -> [pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH]
    result.image = tf.reshape(image, [pgnet.INPUT_SIDE, pgnet.INPUT_SIDE,
                                      DEPTH])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH] of type.float32.
      label: 1-D Tensor of type int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
    images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH] size.
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
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # add a scope to the summary. If shuffle=True, we're training
    # else we're validating
    prefix = 'train' if shuffle else 'validation'
    tf.image_summary(prefix + '/images', images)

    return images, tf.reshape(label_batch, [batch_size])


def train_inputs(csv_path, batch_size):
    """Returns a batch of images from the train dataset.
    Applies random distortion to the examples.

    Args:
        csv_path: path of the cropped pascal dataset
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH size.
        labes: Labels. 1D tensor of [batch_size] size.
    """
    csv_path = os.path.abspath(os.path.expanduser(csv_path)).rstrip("/") + "/"

    # Create a queue that produces the filenames (and other atrributes) to read
    queue = tf.train.string_input_producer([csv_path + "train.csv"])

    # Read examples from the queue
    record = read_cropped_pascal(tf.constant(csv_path), queue)

    # Apply random distortions to the image
    flipped_image = tf.image.random_flip_left_right(record.image)

    # randomize the order of the random distortions
    # thank you:
    # https://stackoverflow.com/questions/37299345/using-if-conditions-inside-a-tensorflow-graph
    def fn1():
        distorted_image = tf.image.random_brightness(flipped_image, max_delta=0.7)
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=0.5, upper=1.)
        return distorted_image
    def fn2():
        distorted_image = tf.image.random_contrast(
            flipped_image, lower=0.5, upper=1.)
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=0.7)
        return distorted_image

    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)
    distorted_image = tf.cond(pred, fn1, fn2)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    print(
        'Filling queue with {} pascal cropped images before starting to train. '
        'This will take a few minutes.'.format(min_queue_examples))
    return _generate_image_and_label_batch(float_image,
                                           record.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=True)


def validation_inputs(csv_path, batch_size):
    """Returns a batch of images from the validation dataset

    Args:
        csv_path: path of the cropped pascal dataset
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE, DEPTH size.
        labes: Labels. 1D tensor of [batch_size] size.
    """

    csv_path = os.path.abspath(os.path.expanduser(csv_path)).rstrip("/") + "/"

    queue = tf.train.string_input_producer([csv_path + "validation.csv"])

    # Read examples from files in the filename queue.
    record = read_cropped_pascal(tf.constant(csv_path), queue)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(record.image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image,
                                           record.label,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=False)
