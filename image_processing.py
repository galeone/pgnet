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


def distort_image(image, input_width, input_height, output_side):
    """Applies random distortion to the image.
    The output image is output_side x output_side x 3
    """

    def random_crop_it():
        """Random crops image, after resizing it to output_side +50 x output_side+50"""
        resized_img = resize_bl(image, output_side + 50)
        return tf.random_crop(image, [output_side, output_side, 3])

    def resize_it():
        """Resize the image using resize_bl"""
        return resize_bl(image, output_side)

    # if input.width >= output.side + 50 and input.heigth >= output.side + 50
    #   resize it to output.side + 50 x output.size + 50 and random crop it
    # else resize it
    increased_output_side = tf.constant(output_side + 50, dtype=tf.int64)
    image = tf.cond(
        tf.logical_and(
            tf.greater_equal(input_width, increased_output_side),
            tf.greater_equal(input_height, increased_output_side)),
        random_crop_it, resize_it)

    # Apply random distortions to the image
    flipped_image = tf.image.random_flip_left_right(image)

    # randomize the order of the random distortions
    def fn1():
        """Applies random brightness, saturation, hue, contrast"""
        distorted_image = tf.image.random_brightness(flipped_image,
                                                     max_delta=0.4)
        distorted_image = tf.image.random_saturation(distorted_image,
                                                     lower=0.5,
                                                     upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2,
                                                   upper=1.2)
        return distorted_image

    def fn2():
        """Applies random brightness, contrast, saturation, hue"""
        distorted_image = tf.image.random_brightness(flipped_image,
                                                     max_delta=0.4)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2,
                                                   upper=1.2)
        distorted_image = tf.image.random_saturation(distorted_image,
                                                     lower=0.5,
                                                     upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)

        return distorted_image

    p_order = tf.random_uniform(shape=[],
                                minval=0.0,
                                maxval=1.0,
                                dtype=tf.float32)
    distorted_image = tf.cond(tf.less(p_order, 0.5), fn1, fn2)
    distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)
    return distorted_image


def train_image(image_path,
                input_width,
                input_height,
                output_side,
                image_type="jpg"):
    """Read the image from image_path.
    Applies distortions and rescale image between -1 and 1
    """

    if image_type == "jpg":
        image = read_image_jpg(image_path, 3)
    else:
        image = read_image_png(image_path, 3)

    image = distort_image(image, input_width, input_height, output_side)
    # rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def eval_image(image_path, output_side, image_type="jpg"):
    """Get an image for evaluation.
    Read the image from image_path.
    Resize the read image to output_side² and rescale values to [-1, 1]
    """
    if image_type == "jpg":
        image = read_image_jpg(image_path, 3)
    else:
        image = read_image_png(image_path, 3)

    image = resize_bl(image, output_side)
    # rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image
