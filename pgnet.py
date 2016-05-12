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



def get(image_):
    """
    @input:
        image_ is a tensor with shape [-1, widht, height, depth]

    As the net goes deeper, increase the number of filters (using power of 2
    in order to optimize GPU performance).

    Always use small size filters because the same effect of a bigger filter can
    be achieved using multiple small filters, that uses less computation resources.

    @returns:
        unscaled_logists: spatial map of output vectors
        [summaries]: list of tensorflow summaries
    """

    kernel_w = kernel_h = 3

    # In the following code, the architecture is defined supposing an input image
    # of at least 184x184 (max value² of the average dimensions of the cropped pascal dataset)
    with tf.variable_scope("l1"):
        num_kernles = 2**6
        kernels = utils.kernels(
            [kernel_w, kernel_h, image_.get_shape()[3].value,
             num_kernles], "kernels")

        bias = utils.bias([num_kernles], "bias")

        # atrous_conv2d:
        # output[b, i, j, k] = sum_{di, dj, q} filters[di, dj, q, k] *
        #      value[b, i + rate * di, j + rate * dj, q]
        # Thus: if rate=1, atrous_conv2d = conv2d with stride [1,1,1,1]

        # In that case, a (3+rate)x(3+rate)x num_kernels is convolved with the input
        # Thus the perception filed of every filter is 5x5xd (rate=2), but the number of parameters is 3x3xd
        # 5 = (filter_h) + (filter_h -1)*(rate -1)
        out1 = tf.nn.relu(
            tf.add(
                tf.nn.atrous_conv2d(image_,
                                    kernels,
                                    rate=2, # means 1 hole
                                    padding='VALID'),
                bias),
            name='relu')

        """ a single image in the out volume is ( (184 -5)/(stride=1) + 1 = 180 )² x 2**6 """

        """ which part of the input volume did not contribute?
        The border contributed only as terms of the weighted sum, but they've never been
        the center of a convolution (because it's impossible to center outside).

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

        # 1 convolution is gone: 3 missing
        out2 = tf.nn.relu(
                tf.add(
                    tf.nn.atrous_conv2d(out1,)))

    with tf.variable_scope("l2"):


    return out, summaries
