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

INPUT_SIDE = 184
INPUT_DEPTH = 3


def padder(input_v, output_v):
    """Extract the borders from input_v.
    The borders size is the difference between output and input height and width.

    If the input depth and the output depth is the same, the padding is made layer by layer.
    eg: padding of layer with depth 1, will be attached to the output layer with depth 1 ecc

    Othwerwise, if the output depth is greather than the input depth (thats the case in
    convolutional neural networks, when output is series of images resulting from
    the convolution of a set of kernels),
    it pads every output layer with the average of the extract border of input.

    input_v: a tensor with [input_batch_size, height, width, input_depth]
    output_v: a tensor with [output_batch_size, reduced_height, reduced_width, output_depth]

    @returns:
        the output volume, padded with the borders of input. Accordingly to the previous description
    """
    input_depth = input_v.get_shape()[3].value
    width = input_v.get_shape()[2].value
    height = input_v.get_shape()[1].value

    output_depth = output_v.get_shape()[3].value
    reduced_width = output_v.get_shape()[2].value
    reduced_height = output_v.get_shape()[1].value

    assert (width - reduced_width) % 2 == 0
    assert (height - reduced_height) % 2 == 0
    assert output_depth >= input_depth

    width_diff = int((width - reduced_width) / 2)
    height_diff = int((height - reduced_height) / 2)

    # every image in the batch have the depth reduced from X to 1 (collpased depth)
    # this single depth is the sum of every depth of the image
    # Or of every depth of the general input volume.
    input_collapsed = tf.reduce_mean(input_v,
                                     reduction_indices=[3],
                                     keep_dims=True)

    # lets make the input depth equal to the output depth
    input_expanded = input_collapsed
    for _ in range(output_depth - 1):
        input_expanded = tf.concat(3, [input_expanded, input_collapsed])

    padding_top = tf.slice(input_expanded, [0, 0, width_diff, 0],
                           [-1, height_diff, reduced_width, -1])
    padding_bottom = tf.slice(input_expanded,
                              [0, height - height_diff, width_diff, 0],
                              [-1, height_diff, reduced_width, -1])

    padded = tf.concat(1, [padding_top, output_v, padding_bottom])

    padding_left = tf.slice(input_expanded, [0, 0, 0, 0], [-1, height,
                                                           width_diff, -1])
    padding_right = tf.slice(input_expanded, [0, 0, width - width_diff, 0],
                             [-1, height, height_diff, -1])

    padded = tf.concat(2, [padding_left, padded, padding_right])
    return padded


def atrous_layer(name, x, kernels_shape, rate):
    """
    Returns the result of:
    ReLU(atrous_conv2d(x, kernels, rate, padding='VALID') + bias).
    Creates kernels, bias and relates summaries.
    
    Args:
        x: 4-D input tensor. shape = [batch, height, width, depth]
        kernels_shape: the shape of W, used in convolution as kernels. [kernel_height, kernel_width, kernel_depth, num_kernels]
        rate: the atrous_conv2d rate parameter
        name: the op name
    """

    num_kernels = kernels_shape[3]
    kernels = utils.kernels(kernels_shape, name + "/kernels")
    bias = utils.bias([num_kernels], name + "/bias")

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
                                rate=rate, # means 1 hole
                                padding='VALID'),
            bias),
        name=name)


def atrous_block(x, kernel_side, rate, num_kernels, exp, name):
    """ atrous block returns 4 atrous convoltion with padded boders as explained below
    params:
        x: [batch_size, height, width, depth]
        kernel_size: we use only square kernels. This is the side length
        rate: atrous_layer rate parameter
        num_kernels: is the number of kernels to learn for the first atrous conv.
            this number crease with an exponential progression across the 4 layer
            Thus: layer1: num_kernels
                layer2: num_nernels *= exp
                layer3: num_lernels *= exp
            num_kernels should be a power of exp, if you want exponential progression.
        exp: see num_kernels
        name: is the block name

    A single image in the out volume is ( (184 -5)/(stride=1) + 1 = 180 )² x 2**6

    Which part of the input volume did not contribute?
    The border contributed only as terms of the weighted sum (where the weight is
    the learned filter value in that position) but they've never been
    the center of a convolution (because it's impossible).

    IDEA: pad the output volume with the previous volume borders. In that way,
    we don't discard the borders contribution along the layer.
    We don't insert zeros ad padding but we use the previous layers border ad padding.

    A central pixel (a pixel that can be centered in the convolution window) concours for
    4 times to the formation of the convolution result.

    A border pixel only one time.
    Thus: IDEA:
    pad with previous input and convolve 4 times (every time with the previous layer padding and the new result)
    The 4 iteration we do not pad the output.

    It's like a residual network, but only for borders :D
    """

    with tf.name_scope("atrous_block"):
        kernels_shape = [kernel_side, kernel_side, x.get_shape()[3].value,
                         num_kernels]
        conv1 = atrous_layer(name + "/conv1", x, kernels_shape, rate)

        # 1 convolution is gone: 3 missing

        # 2
        padded = padder(x, conv1)
        prev_num_kernels = num_kernels
        num_kernels *= exp
        kernels_shape[2] = prev_num_kernels
        kernels_shape[3] = num_kernels
        conv2 = atrous_layer(name + "/conv2", padded, kernels_shape, 2)
        # 3
        padded = padder(x, conv2)
        prev_num_kernels = num_kernels
        num_kernels *= exp
        kernels_shape[2] = prev_num_kernels
        kernels_shape[3] = num_kernels
        conv3 = atrous_layer(name + "/conv3", padded, kernels_shape, 2)
        # 4
        padded = padder(x, conv3)
        prev_num_kernels = num_kernels
        num_kernels *= exp
        kernels_shape[2] = prev_num_kernels
        kernels_shape[3] = num_kernels
        conv4 = atrous_layer(name + "/conv4", padded, kernels_shape, 2)
        return conv4


def get(image_, keep_prob=1.0, num_class=20):
    """
    @input:
        image_ is a tensor with shape [-1, widht, height, depth]
        keep_prob: dropput probability. Set it to something < 1.0 during train
        num_class: number of class to claffify.

    As the net goes deeper, increase the number of filters (using power of 2
    in order to optimize GPU performance).

    Always use small size filters because the same effect of a bigger filter can
    be achieved using multiple small filters, that uses less computation resources.

    @returns:
        unscaled_logists: spatial map of output vectors
    """

    kernel_side = 3

    # In the following code, the architecture is defined supposing an input image
    # of at least 184x184 (max value² of the average dimensions of the cropped pascal dataset)
    with tf.variable_scope("l1"):
        num_kernels = 2**6
        # at the end of block1, num_kernels has increased to: 2**(6+4 - 1) = 2**9 = 512
        block1 = atrous_block(
            image_, kernel_side,
            2, num_kernels,
            exp=2, name="block1")
        num_kernels = 2**9
    #output: 180x180x512
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
    pool1 = atrous_layer("atrous_pooling_1", block1,
                         [kernel_side, kernel_side, num_kernels,
                          num_kernels], 45)
    #output: 90x90x512
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
    with tf.variable_scope("l2"):
        block2 = atrous_block(
            pool1, kernel_side,
            2, num_kernels,
            exp=1, name="block2")
        #output: lxlx512, l = (90 -5)/(stride=1) + 1 = 86
        #output: 86x86x512
        print(block2)

    # reduce 4 times the input volume
    # 86/4 = 22
    # 22 = 86 - filter_size + 1 -> filter_size = 86-22+1 = 65
    # new filter side = side + (side -1)*(rate -1)
    # 65 = 3 + (3 - 1)*(rate -1) -> rate = 64/2 = 32

    # 86/2 = 43 -> is odd, use 42
    # 42 = 86 - filter_side + 1 -> filter_side = 86-42 +1 = 45
    # new filter side = side + (side -1)*(rate -1)
    # 45 = 3 + (3 - 1)*(rate -1) -> rate = 42/2 = 22
    pool2 = atrous_layer("atrous_pooling_2", block2,
                         [kernel_side, kernel_side, num_kernels,
                          num_kernels], 22)
    #output: 42x42x512
    print(pool2)

    with tf.variable_scope("l3"):
        block3 = atrous_block(
            pool2, kernel_side,
            2, num_kernels,
            exp=1, name="block3")

        # l = (42-5) +1 = 38
        #output: 38x38x512
    print(block3)

    # 38/2 = 19
    # 19 = 38 - filter_side + 1 -> filter_side = 38-19 +1 = 20
    # new filter side = side + (side -1)*(rate -1)
    # 20 = 3 + (3 - 1)*(rate -1) -> rate = 19/2 = 9.5 -> 10 -> 18x18
    pool3 = atrous_layer("atrous_pooling_3", block3,
                         [kernel_side, kernel_side, num_kernels,
                          num_kernels], 10)
    #output: 18x18x512
    print(pool3)

    # fully convolutional layer
    # take the 85x85x512 input and project it to a 1x1x1024 dim space
    with tf.variable_scope('fc1'):
        W_fc1 = utils.kernels([18, 18, 512, 1024], "W")
        b_fc1 = utils.bias([1024], "b")

        h_fc1 = tf.nn.relu(
            tf.add(
                tf.nn.conv2d(pool3, W_fc1, [1, 1, 1, 1],
                             padding='VALID'),
                b_fc1),
            name='h')
        # output: 1x1x1024
        print(h_fc1)

    with tf.variable_scope('dropout') as scope:
        dropoutput = tf.nn.dropout(h_fc1, keep_prob, name=scope.name)
        print(dropoutput)
        # output: 1x1x1024

        # softmax(WX + n)
    with tf.variable_scope('softmax_linear') as scope:
        # convert this 1024 featuers to num_class (usually 20)
        W_fc2 = utils.kernels([1, 1, 1024, num_class], "W")
        b_fc2 = utils.bias([num_class], "b")
        out = tf.add(
            tf.nn.conv2d(dropoutput,
                         W_fc2, [1, 1, 1, 1],
                         padding='VALID'),
            b_fc2,
            name=scope.name)
        # outout: 1x1x20 if the input has been properly scaled
        # otherwise is a map
        print(out)

    return out
