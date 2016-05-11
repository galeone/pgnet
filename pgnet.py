import tensorflow as tf
import utils
"""
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
    summaries = []

    with tf.variable_scope("l1") as l1:
        num_kernles = 2**6
        kernels, summary = utils.kernels(
            [kernel_w, kernel_h, image_.get_shape()[3].value,
             num_kernles], "kernel_bank_1")
        summaries.append(summary)

        biases, summary = utils.weight([num_kernles], "kernels_bias")
        summaries.append(summary)

        # atrous_conv2d:
        # output[b, i, j, k] = sum_{di, dj, q} filters[di, dj, q, k] *
        #      value[b, i + rate * di, j + rate * dj, q]
        # Thus: if rate=1, atrous_conv2d = conv2d with stride [1,1,1,1]
        out = tf.nn.relu(
            tf.add(
                tf.nn.atrous_conv2d(image_,
                                    kernels,
                                    rate=2, # means 1 hole
                                    padding='VALID',
                                    name='atrous_conv2d'),
                biases),
            name='out')

    return out, summaries
