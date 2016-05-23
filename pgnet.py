import tensorflow as tf
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
NUM_EPOCHS_PER_DECAY = 5  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-2  # Initial learning rate.


def atrous_layer(x, atrous_kernel_shape, rate, padding):
    """
    Returns the result of:
    ReLU(atrous_conv2d(x, kernels, rate, padding=padding) + bias).
    Creates kernels (name=kernel), bias (name=bias) and relates summaries.
    
    Args:
        x: 4-D input tensor. shape = [batch, height, width, depth]
        atrous_kernel_shape: the shape of W, used in convolution as kernels. [kernel_height, kernel_width, kernel_depth, num_kernels]
        rate: the atrous_conv2d rate parameter
        name: the op name
        padding; "VALID" | "SAME"
    """

    num_kernels = atrous_kernel_shape[3]

    kernels = utils.kernels(atrous_kernel_shape, "kernels")
    bias = utils.bias([num_kernels], "bias")

    # atrous_conv2d:
    # output[b, i, j, k] = sum_{di, dj, q} filters[di, dj, q, k] *
    #      value[b, i + rate * di, j + rate * dj, q]
    # Thus: if rate=1, atrous_conv2d = conv2d with stride [1,1,1,1]

    # Supposing a 3x3 filter:
    # In that case, a (3+rate)x(3+rate)x num_kernels is convolved with the input
    # Thus the perception filed of every filter is 5x5xd (rate=2), but the number of parameters is 3x3xd
    # 5 = (filter_h) + (filter_h -1)*(rate -1)
    return tf.nn.relu(
        tf.add(
            tf.nn.atrous_conv2d(x,
                                kernels,
                                rate=rate, # rate = 2 means 1 hole
                                padding=padding),
            bias),
        name="out")


def eq_conv(x, atrous_kernel_side, num_kernels, rate, padding=False):
    """Pads the input with the right amount of zeros.
    Atrous convolve the padded input. In that way every pixel of the input image
    will contribute on average.
    Than extracts the average contributions of every kernel side x kernel side
    location in the resulting image, using a stride of 1, and use the average poooling
    to make the contribution of every location equal.
    Pads the results with zeros of required
    """

    with tf.variable_scope("eq_conv"):
        atrous_kernel_shape = [atrous_kernel_side, atrous_kernel_side,
                               x.get_shape()[3].value, num_kernels]
        # extract "real" kernel side of the atrous filter
        real_kernel_side = atrous_kernel_side + (atrous_kernel_side - 1) * (
            rate - 1)
        # pad the input with the right amount of padding
        pad_amount = int((real_kernel_side - 1) / 2)
        input_padded = tf.pad(x,
                              [[0, 0], [pad_amount, pad_amount],
                               [pad_amount, pad_amount], [0, 0]],
                              name="input_padded")
        print(input_padded)

        # convolve padded input with learned filters.
        # using the padded input we can handle border pixesl
        conv_contribution = atrous_layer(input_padded,
                                         atrous_kernel_shape,
                                         rate,
                                         padding="VALID")
        print(conv_contribution)
        """
        # lets make the pixels contribut equally, using average pooling
        # with a stride of 1 and a ksize equals to the kernel size in order
        # to reside the contribution output to the original convolution size
        # eg: the result of a convolution without the input padded
        eq = tf.nn.avg_pool(conv_contribution,
                            ksize=[1, real_kernel_side, real_kernel_side, 1],
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="eq")
        """
        eq = conv_contribution
        print(eq)

        if padding:
            top_bottom = int(
                (x.get_shape()[1].value - eq.get_shape()[1].value) / 2)
            left_right = int(
                (x.get_shape()[2].value - eq.get_shape()[2].value) / 2)
            eq = tf.pad(eq,
                        [[0, 0], [top_bottom, top_bottom],
                         [left_right, left_right], [0, 0]],
                        name="padded_eq")

        print(eq)
    return eq


def atrous_block(x, atrous_kernel_side, rate, num_kernels, exp):
    """ atrous block returns the result of 4 atrous convoltion, using the eq_conv
    The output of eq_conv is zero paded for the first 3 convolution. The last one is not padded.

    params:
        x: [batch_size, height, width, depth]
        atrous_kernel_size: we use only square kernels. This is the side length
        rate: atrous_layer rate parameter
        num_kernels: is the number of kernels to learn for the first atrous conv.
            this number crease with an exponential progression across the 4 layer, skipping ne
            Thus:
                layer1, layer2: num_nernels
                layer3, layer4: num_lernels *= exp
            num_kernels should be a power of exp, if you want exponential progression.
        exp: see num_kernels
    """
    with tf.variable_scope("conv1"):
        conv1 = eq_conv(x, atrous_kernel_side, num_kernels, rate, padding=True)

    with tf.variable_scope("conv2"):
        conv2 = eq_conv(
            conv1, atrous_kernel_side,
            num_kernels,
            rate, padding=True)

    num_kernels *= exp
    with tf.variable_scope("conv3"):
        conv3 = eq_conv(
            conv2, atrous_kernel_side,
            num_kernels,
            rate, padding=True)

    with tf.variable_scope("conv4"):
        conv4 = eq_conv(
            conv3, atrous_kernel_side,
            num_kernels,
            rate, padding=False)

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

    atrous_kernel_side = 3

    # In the following code, the architecture is defined supposing an input image
    # of at least 184x184 (max value² of the average dimensions of the cropped pascal dataset)
    with tf.variable_scope("block1"):
        num_kernels = 2**5
        # at the end of block1, num_kernels has increased to: 2**(6+4 - 1) = 2**9 = 512
        block1 = atrous_block(
            image_, atrous_kernel_side,
            2, num_kernels, exp=2)
        num_kernels *= 2
    #output: 184x184x512
    print(block1)

    # now that the border contributed 4 times
    # we can reduce dimensionality of the block1 output.
    # we don't use max pool, but we increase the rate parameter of the atrous convolution
    # in order to preserve the spatial relations that extists between input and output volumes
    # I want to hald the dimension of the input volume (max pooling like).
    # 90 = 180 - filer_size + 1 -> filer_size = 180-90 +1 = 91
    # new filter side = side + (side - 1)*(rate -1)
    # 91 = 3 + (3 - 1)*(rate -1) -> rate = 90/2 = 45
    # Thus the new widht and height is: 180 - 2*45 = 90
    # That the spatial extent of a polling with a 2x2 window with a stride of 2.
    # The diffence is that the pooling is not learneable, the filter is.
    with tf.variable_scope("pool1"):
        pool1 = tf.nn.max_pool(block1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID")
    #output: 92x92x512
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
        block2 = atrous_block(pool1, atrous_kernel_side, 2, num_kernels, exp=2)
        num_kernels *= 2
        #output: lxlx512, l = (92 -5)/(stride=1) + 1 = 88
        #output: 88x88x512
        print(block2)
        print(num_kernels)

    # reduce 4 times the input volume
    # 86/4 = 22
    # 22 = 86 - filter_size + 1 -> filter_size = 86-22+1 = 65
    # new filter side = side + (side -1)*(rate -1)
    # 65 = 3 + (3 - 1)*(rate -1) -> rate = 64/2 = 32

    # 86/2 = 43 -> is odd, use 42
    # 42 = 86 - filter_side + 1 -> filter_side = 86-42 +1 = 45
    # new filter side = side + (side -1)*(rate -1)
    # 45 = 3 + (3 - 1)*(rate -1) -> rate = 42/2 = 22
    with tf.variable_scope("pool2"):
        pool2 = tf.nn.max_pool(block2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID")
    #output: 44x44x512
    print(pool2)

    with tf.variable_scope("block3"):
        block3 = atrous_block(pool2, atrous_kernel_side, 2, num_kernels, exp=2)
        num_kernels *= 2
        print(num_kernels)
        # l = (44-5) +1 = 40
        #output: 40x40x512
    print(block3)

    # 40/2 = 20
    with tf.variable_scope("pool3"):
        pool3 = tf.nn.max_pool(block3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding="VALID")

    #output: 23x23x512
    print(pool3)

    # fully convolutional layer
    # take the 20x85x512 input and project it to a 1x1xNUM_NEURONS dim space
    NUM_NEURONS = 1024
    with tf.variable_scope("fc1"):
        W_fc1 = utils.kernels([23, 23, num_kernels, NUM_NEURONS], "W")
        b_fc1 = utils.bias([NUM_NEURONS], "b")

        h_fc1 = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(pool3, W_fc1, [1, 1, 1, 1],
                             padding="VALID"),
                b_fc1),
            name="h")
        # output: 1x1xNUM_NEURONS
        print(h_fc1)

    with tf.variable_scope("dropout"):
        dropoutput = tf.nn.dropout(h_fc1, keep_prob, name="out")
        print(dropoutput)
        # output: 1x1xNUM_NEURONS

        # softmax(WX + n)
    with tf.variable_scope("softmax_linear"):
        # convert this NUM_NEURONS featuers to NUM_CLASS
        W_fc2 = utils.kernels([1, 1, NUM_NEURONS, NUM_CLASS], "W")
        b_fc2 = utils.bias([NUM_CLASS], "b")
        out = tf.add(
            tf.nn.conv2d(dropoutput,
                         W_fc2, [1, 1, 1, 1],
                         padding="VALID"),
            b_fc2,
            name="out")
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
    with tf.variable_scope("train"):
        optimizer = tf.train.AdadeltaOptimizer(INITIAL_LEARNING_RATE)
        # minimizes loss and increments global_step by 1
        minimizer = optimizer.minimize(loss_op, global_step=global_step)

    return minimizer
