#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""utils contains utilities to work with tensorflow
and make the summary generation easier """

import math
import tensorflow as tf


def print_graph_ops(graph):
    """used in debug: print graph operations"""
    ops = graph.get_operations()
    for op in ops:
        print("Op name: %s" % op.name)
        for k in op.inputs:
            print("\tin: %s %s" % (k.name, k.get_shape()))
        for k in op.outputs:
            print("\tout: %s %s" % (k.name, k.get_shape()))
        print("")


def variable_summaries(var, name):
    """ variables_summaries creates the following summaries for var, with name.
    1. mean
    2. stddev
    3. min
    4. max
    And the histogram of var.
    Returns the defined summaries in a list.
    The returned value can be discarted if the function has been called inside the default graph.

    Remeber to call tf.merge_all_summaries() before tf.initialize_all_variables()
    """

    with tf.name_scope("summaries"):
        summaries = []
        mean = tf.reduce_mean(var)
        summaries.append(tf.scalar_summary('mean/' + name, mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        summaries.append(tf.scalar_summary('sttdev/' + name, stddev))
        summaries.append(tf.scalar_summary('max/' + name, tf.reduce_max(var)))
        summaries.append(tf.scalar_summary('min/' + name, tf.reduce_min(var)))
        summaries.append(tf.histogram_summary(name, var))
        return summaries


def weight(shape, name):
    """ weight returns a tensor with the requested shape, initialized with
    the xavier initializer
    Creates baisc summaries too.
    Returns the weight tensor"""
    w = tf.get_variable(name,
                        shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    _ = variable_summaries(w, w.name)
    return w


def kernels(shape, name):
    """ kernels create and return a weight with the required shape
    The main difference with weight, is that kernels returns the summary
    for the learned filters visualization if the weight depth is 1, 2 or 3.
    shape should be in the form [ Y, X, Depth, NumKernels ]
    """
    w = weight(shape, name)

    if shape[2] in (1, 3, 4):
        with tf.name_scope("summaries"):
            num_kernels = shape[3]
            depth = shape[2]
            max_images = int(num_kernels / depth)
            tf.image_summary(name,
                             tf.reshape(w, [num_kernels, shape[0], shape[1],
                                            depth]),
                             max_images=max_images)
    return w


def bias(shape, name, init_val=0.0):
    """ bias returns a tensor with the requested shape, initialized with init_val.
    Creates summaries too.
    Returns the bias"""
    b = tf.get_variable(
        name, shape, initializer=tf.constant_initializer(init_val))
    _ = variable_summaries(b, b.name)
    return b


def kernels_on_grid_summary(kernel, name):
    """ Returns the Summary with kernel filters displayed in a single grid
    Visualize conv. features as an image (mostly for the 1st layer).
    Args:
        kernel: tensor of shape [Y, X, NumChannels, NumKernels]
        name: the name displayed in tensorboard
    """
    #TODO: fixme

    pad = 1
    kernel_height = kernel.get_shape()[0].value + pad
    kernel_width = kernel.get_shape()[1].value + pad
    depth = kernel.get_shape()[2].value
    num_kernels = kernel.get_shape()[3].value
    num_filters = int(num_kernels / depth)

    square_side = math.ceil(math.sqrt(num_kernels))
    grid_height = square_side * kernel_height + 1
    grid_width = square_side * kernel_width + 1

    # split kernel in num_filters filter and put it into the grid
    # pad the extracted filter
    filters = tf.split(3, num_filters, kernel)
    y_pos, x_pos = 0, 0

    # list of tensors
    cells = []
    for inner_filter in filters:
        filter_3d = tf.squeeze(inner_filter, [3])
        # add padding
        padding = tf.constant([[pad, 0], [pad, 0], [0, 0]])
        filter_3d = tf.pad(filter_3d, padding)

        before_padding = tf.constant([[y_pos, 0], [x_pos, 0], [0, 0]])

        bottom_padding = grid_width - y_pos - kernel_width - 1
        right_padding = grid_height - x_pos - kernel_height - 1
        after_paddng = tf.constant([[bottom_padding, 1], [right_padding, 1],
                                    [0, 0]])

        cell = tf.pad(filter_3d, before_padding)
        cells.append(tf.pad(cell, after_paddng))

        if right_padding == 0:
            # move along y
            y_pos += kernel_height
            # reset x position
            x_pos = 0
        else:
            # move along x
            x_pos += kernel_height

    grid = tf.reshape(tf.add_n(cells), [1, grid_width, grid_height, depth])
    return tf.image_summary(name, grid, max_images=1)


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
