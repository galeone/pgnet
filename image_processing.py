#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utilities for image processing"""

import tensorflow as tf


def resize_bl(image, side):
    """Returns the image, resized with bilinear interpolation to: side x side x depth
    Input:
        image: 3d tensor width shape [side, side, depth]
    """

    #reshape to a 4-d tensor (required to resize)
    image = tf.expand_dims(image, 0)
    # now image is 4-D float32 tensor: [1, side, side, image.depth]
    image = tf.image.resize_bilinear(image, [side, side], align_corners=False)
    # remove the 1st dimension
    image = tf.squeeze(image, [0])
    return image


def read_image_jpg(image_path, depth=3):
    """Reads the image from image_path (tf.string tensor) [jpg image].
    Cast the result to float32.
    Reuturn:
        the decoded jpeg image, casted to float32
    """
    return tf.image.convert_image_dtype(
        tf.image.decode_jpeg(
            tf.read_file(image_path), channels=depth),
        dtype=tf.float32)


def read_image_png(image_path, depth=3):
    """Reads the image from image_path (tf.string tensor) [jpg image].
    Cast the result to float32.
    Reuturn:
        the decoded jpeg image, casted to float32
    """
    return tf.image.convert_image_dtype(
        tf.image.decode_png(
            tf.read_file(image_path), channels=depth),
        dtype=tf.float32)


def distort_image(image, input_width, input_height, output_side, output_depth):
    """Applies random distortion to the image.
    The output image is output_side x output_side x output_depth
    """

    def random_crop_it():
        """Random crops image"""
        return tf.random_crop(image, [output_side, output_side, output_depth])

    def resize_it():
        """Resize the image using resize_bl"""
        return resize_bl(image, output_side)

    output_side_const = tf.constant(output_side, dtype=tf.int64)

    # if image.input_width >= side and image.input_height >= input side:
    #   random crop it with probability p
    # else resize it
    p_crop = tf.random_uniform(shape=[],
                               minval=0.0,
                               maxval=1.0,
                               dtype=tf.float32)
    image = tf.cond(
        tf.logical_and(
            tf.less(p_crop, 0.5), tf.logical_and(
                tf.greater_equal(input_width, output_side_const),
                tf.greater_equal(input_height, output_side_const))),
        random_crop_it, resize_it)

    # Apply random distortions to the image
    flipped_image = tf.image.random_flip_left_right(image)

    # randomize the order of the random distortions
    # thanks to:
    # https://stackoverflow.com/questions/37299345/using-if-conditions-inside-a-tensorflow-graph
    def fn1():
        """Applies random brightness and random contrast"""
        distorted_image = tf.image.random_brightness(flipped_image,
                                                     max_delta=0.4)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2,
                                                   upper=1.2)
        return distorted_image

    def fn2():
        """Applies random constrast and random brightness"""
        distorted_image = tf.image.random_contrast(flipped_image,
                                                   lower=0.2,
                                                   upper=1.2)
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=0.4)
        return distorted_image

    p_order = tf.random_uniform(shape=[],
                                minval=0.0,
                                maxval=1.0,
                                dtype=tf.float32)
    distorted_image = tf.cond(tf.less(p_order, 0.5), fn1, fn2)
    return distorted_image
