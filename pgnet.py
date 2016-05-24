import tensorflow as tf
import pascal_input
import utils
"""
The model is fully convolutional, thus it accepts batch of images of any size and produces
a spatial map of vector.
The degenerate case is when the input size have the same size of the train images. In that case
the output is a (batchsize x)1x1x<num_classes> tensor

Conventions:
    var_: placeholder
"""

# dataset constants
INPUT_SIDE = 184
INPUT_DEPTH = 3
NUM_CLASS = 20

# train constants
BATCH_SIZE = 32
NUM_EPOCHS_PER_DECAY = 2  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4  # Initial learning rate.

# number of neurons in the last "fully connected" (1x1 conv) layer
NUM_NEURONS = 1024


def conv_layer(input_x, kernel_shape, padding):
    """
    Returns the result of:
    ReLU(conv2d(x, kernels, padding=padding) + bias).
    Creates kernels (name=kernel), bias (name=bias) and relates summaries.
    
    Args:
        x: 4-D input tensor. shape = [batch, height, width, depth]
        kernel_shape: the shape of W, used in convolution as kernels. [kernel_height, kernel_width, kernel_depth, num_kernels]
        name: the op name
        padding; "VALID" | "SAME"
    """

    num_kernels = kernel_shape[3]

    kernels = utils.kernels(kernel_shape, "kernels")
    bias = utils.bias([num_kernels], "bias")

    return tf.nn.relu(
        tf.add(
            tf.nn.conv2d(input_x,
                         kernels,
                         strides=[1, 1, 1, 1],
                         padding=padding),
            bias),
        name="out")


def eq_conv_layer(input_x, kernel_side, num_kernels):
    """Pads the input with the right amount of zeros.
    Convolve the padded input. In that way every pixel of the input image
    will contribute on average.
    Output WxH = input WxH
    """

    with tf.variable_scope("eq_conv_layer"):
        kernel_shape = [kernel_side, kernel_side, input_x.get_shape()[3].value,
                        num_kernels]

        # pad the input with the right amount of padding
        pad_amount = int((kernel_side - 1) / 2)
        input_padded = tf.pad(input_x,
                              [[0, 0], [pad_amount, pad_amount],
                               [pad_amount, pad_amount], [0, 0]],
                              name="input_padded")
        print(input_padded)
        return conv_layer(input_padded, kernel_shape, padding="VALID")


def block(input_x, kernel_side, num_kernels, exp):
    """ block returns the result of 4 convolution, using the eq_conv_layer.
    The first thw layers have num_kernels kernels, the last two num_kernels*exp

    params:
        input_x: [batch_size, height, width, depth]
        _kernel_size: we use only square kernels. This is the side length
        num_kernels: is the number of kernels to learn for the first conv.
            this number crease with an exponential progression across the 4 layer, skipping ne
            Thus:
                layer1, layer2: num_nernels
                layer3, layer4: num_lernels *= exp
            num_kernels should be a power of exp, if you want exponential progression.
        exp: see num_kernels
    """
    with tf.variable_scope("conv1"):
        conv1 = eq_conv_layer(input_x, kernel_side, num_kernels)

    with tf.variable_scope("conv2"):
        conv2 = eq_conv_layer(conv1, kernel_side, num_kernels)

    num_kernels *= exp
    with tf.variable_scope("conv3"):
        conv3 = eq_conv_layer(conv2, kernel_side, num_kernels)

    with tf.variable_scope("conv4"):
        conv4 = eq_conv_layer(conv3, kernel_side, num_kernels)

    return conv4


def get(image_, keep_prob=1.0):
    """
    @input:
        image_ is a tensor with shape [-1, widht, height, depth]
        keep_prob: dropput probability. Set it to something < 1.0 during train

    As the net goes deeper, increase the number of filters (using power of 2
    in order to optimize GPU performance).

    Always use small size filters because the same effect of a bigger filter can
    be achieved using multiple small filters, that uses less computation resources.

    @returns:
        softmax_linear: spatial map of output vectors (unscaled)
    """
    print(image_)

    kernel_side = 3

    # In the following code, the architecture is defined supposing an input image
    # of at least 184x184 (max value² of the average dimensions of the cropped pascal dataset)
    with tf.variable_scope("block1"):
        num_kernels = 2**5
        block1 = block(image_, kernel_side, num_kernels, exp=2)
        num_kernels *= 2
    #output: 184x184x64
    print(block1)

    with tf.variable_scope("pool1"):
        pool1 = tf.nn.max_pool(block1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID")
    #output: 92x92x64
    print(pool1)

    # normalization is useless
    """
    CS231n: http://cs231n.github.io/convolutional-networks/
    Many types of normalization layers have been proposed for use in ConvNet architectures, sometimes
    with the intentions of implementing inhibition schemes observed in the biological brain.
    However, these layers have recently fallen out of favor because in practice their contribution has
    been shown to be minimal, if any.
    For various types of normalizations, see the discussion in Alex Krizhevsky's cuda-convnet library API.
    """

    # repeat the l1, using pool1 as input. Do not incrase the number of learned filter
    # Preserve input depth
    with tf.variable_scope("block2"):
        block2 = block(pool1, kernel_side, num_kernels, exp=2)
        num_kernels *= 2
        #output: 92x92x128
        print(block2)

    # reduce 2 times the input volume
    # 92/2 = 46
    with tf.variable_scope("pool2"):
        pool2 = tf.nn.max_pool(block2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID")
    #output: 46x46x512
    print(pool2)

    with tf.variable_scope("block3"):
        block3 = block(pool2, kernel_side, num_kernels, exp=2)
        num_kernels *= 2
        #output: 46x46x512
        print(block3)

    # 46/2 = 23
    with tf.variable_scope("pool3"):
        pool3 = tf.nn.max_pool(block3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID")
        #output: 23x23x512
        print(pool3)

    # fully convolutional layer
    # take the 23x23x512 input and project it to a 1x1xNUM_NEURONS dim space
    with tf.variable_scope("fc1"):
        fc1 = conv_layer(pool3, [23, 23, num_kernels, NUM_NEURONS],
                         padding="VALID")
        # output: 1x1xNUM_NEURONS
        dropout = tf.nn.dropout(fc1, keep_prob, name="dropout")
        print(dropout)
        # output: 1x1xNUM_NEURONS

    with tf.variable_scope("softmax_linear"):
        out = conv_layer(dropout, [1, 1, NUM_NEURONS, NUM_CLASS],
                         padding="VALID")
        # output: (BATCH_SIZE)1x1xNUM_CLASS if the input has been properly scaled
        # otherwise is a map
        print(out)

    return out


def loss(logits, labels):
    """
    Args:
        logits: Logits from get().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [batch_size]

    Returns:
        Loss tensor of type float.
    """

    with tf.variable_scope("loss"):
        # reshape logits to a vector of NUM_CLASS elements
        # -1 = every batch size
        logits = tf.reshape(logits, [-1, NUM_CLASS])

        labels = tf.cast(labels, tf.int64)

        # cross_entropy across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name="cross_entropy_per_example")

        mean_cross_entropy = tf.reduce_mean(cross_entropy,
                                            name="mean_cross_entropy")
        tf.scalar_summary("loss/mean_cross_entropy", mean_cross_entropy)

    return mean_cross_entropy


def train(loss_op, global_step):
    """
    Creates an Optimizer.
    Args:
        loss_op: loss from loss()
        global_step: integer variable counting the numer of traning steps processed

    Returns:
        train_op: of for training
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = pascal_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    with tf.variable_scope("train"):
        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                   global_step,
                                                   decay_steps,
                                                   LEARNING_RATE_DECAY_FACTOR,
                                                   staircase=True)

        tf.scalar_summary('learning_rate', learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        # minimizes loss and increments global_step by 1
        minimizer = optimizer.minimize(loss_op, global_step=global_step)

    return minimizer
