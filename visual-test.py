import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import utils

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

    atrous_kernel_shape = [atrous_kernel_side, atrous_kernel_side,
                           x.get_shape()[3].value, num_kernels]


    with tf.variable_scope("eq_conv") as scope:

        # extract "real" kernel side of the atrous filter
        real_kernel_side = atrous_kernel_side + (atrous_kernel_side - 1) * (
            rate - 1)

        std_conv = atrous_layer(x, atrous_kernel_shape, rate, padding="VALID")

        # pad the input with the right amount of padding
        pad_amount = int((real_kernel_side - 1) / 2)
        input_padded = tf.pad(x,
                              [[0, 0], [pad_amount, pad_amount],
                               [pad_amount, pad_amount], [0, 0]],
                              name="input_padded")
        print(input_padded)

        #reuse var
        scope.reuse_variables()

        # convolve padded input with learned filters.
        # using the padded input we can handle border pixesl
        conv_contribution = atrous_layer(input_padded,
                                         atrous_kernel_shape,
                                         rate,
                                         padding="VALID")
        print(conv_contribution)

        # lets make the pixels contribut equally, using average pooling
        # with a stride of 1 and a ksize equals to the kernel size in order
        # to reside the contribution output to the original convolution size
        # eg: the result of a convolution without the input padded
        eq = tf.nn.avg_pool(conv_contribution,
                            ksize=[1, real_kernel_side, real_kernel_side, 1],
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="eq")
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
    return conv_contribution, eq, std_conv

image_path = "/data/PASCAL_2012_cropped/2010_003078.jpg"
image = cv2.imread(image_path)
image1 = cv2.imread(image_path)

images_ = tf.placeholder(tf.float32, [None, image.shape[0], image.shape[1], 3])

kernels = tf.get_variable("kernels", [3, 3, 3, 3],
                          initializer=tf.random_normal_initializer())

reshaped_input_tensor = tf.reshape(
    tf.pack([tf.cast(image, tf.float32), tf.cast(image1, tf.float32)]),
    [-1, image.shape[0], image.shape[1], image.shape[2]])
print(reshaped_input_tensor)

conv_contrib_op, eq_op, std_conv_op = eq_conv(reshaped_input_tensor, 3, 3, 2)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(image.shape)
    print(kernels.get_shape())
    reshaped_input = reshaped_input_tensor.eval()
    cv2.imshow("original", tf.cast(reshaped_input[0], tf.uint8).eval())

    #always remember to cast to tf.uint8 if you want to display the result with opencv
    conv_contrib, eq, std_conv = sess.run([conv_contrib_op, eq_op,std_conv_op])
    cv2.imshow("conv no pad", tf.cast(std_conv[0], tf.uint8).eval())
    print("conv no pad size")
    print(std_conv[0].shape)
    cv2.imshow("conv + pad", tf.cast(conv_contrib[0], tf.uint8).eval())
    print("conv + pad size")
    print(conv_contrib[0].shape)
    cv2.imshow("eq conv (avg_pool)", tf.cast(eq[0], tf.uint8).eval())
    print("eq_convg avg-pool size")
    print(eq[0].shape)

    cv2.waitKey(0)
