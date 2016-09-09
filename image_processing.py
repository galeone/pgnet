#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utilities for image processing"""

import math
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


def read_image(image_path, channel, image_type):
    """Wrapper around read_image_{jpg,png}"""
    if image_type == "jpg":
        image = read_image_jpg(image_path, 3)
    else:
        image = read_image_png(image_path, 3)
    return image


def distort_image(image, input_width, input_height, output_side):
    """Applies random distortion to the image.
    The output image is output_side x output_side x 3
    """

    def random_crop_it():
        """Random crops image, after resizing it to output_side +10 x output_side+10"""
        resized_img = resize_bl(image, output_side + 10)
        return tf.random_crop(resized_img, [output_side, output_side, 3])

    def resize_it():
        """Resize the image using resize_bl"""
        return resize_bl(image, output_side)

    # if input.width >= output.side + 10 and input.heigth >= output.side + 10
    #   resize it to output.side + 10 x output.size + 10 and random crop it
    # else resize it
    increased_output_side = tf.constant(output_side + 10, dtype=tf.int64)
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
        distorted_image = tf.image.random_brightness(
            flipped_image, max_delta=32. / 255.)
        distorted_image = tf.image.random_saturation(
            distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=0.5, upper=1.5)
        return distorted_image

    def fn2():
        """Applies random brightness, contrast, saturation, hue"""
        distorted_image = tf.image.random_brightness(
            flipped_image, max_delta=32. / 255.)
        distorted_image = tf.image.random_contrast(
            distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_saturation(
            distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)

        return distorted_image

    p_order = tf.random_uniform(
        shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)
    distorted_image = tf.cond(tf.less(p_order, 0.5), fn1, fn2)
    distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)
    return distorted_image


def zm_mp(image):
    """Keeps an image with values in [0, 1). Normalizes it in order to be
    centered and have zero mean.
    Normalizes its values in range [-1, 1]."""
    image = tf.image.per_image_whitening(image)

    # rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def train_image(image_path,
                input_width,
                input_height,
                output_side,
                image_type="jpg"):
    """Read the image from image_path.
    Applies distortions and rescale image between -1 and 1
    """
    image = read_image(image_path, 3, image_type)
    image = distort_image(image, input_width, input_height, output_side)
    image = zm_mp(image)
    return image


def eval_image(image_path, output_side, image_type="jpg"):
    """Get an image for evaluation.
    Read the image from image_path.
    Resize the read image to output_side² and rescale values to [-1, 1]
    """

    image = read_image(image_path, 3, image_type)
    image = resize_bl(image, output_side)
    image = zm_mp(image)
    return image


def get_original_and_processed_image(image_path, side, image_type="jpg"):
    """Return the original image as read from image_path and the image splitted as a batch tensor.
    Args:
        image_path: image path
        side: image side
        image_type: image type
    Returns:
        original_image, eval_image
        where original image is a tensor in the format [widht, height 3]
        eval_image is a 4-D tensor, with shape [1, w, h, 3]"""

    original_image = read_image(image_path, 3, image_type)

    normalized_patches = []

    # the whole image resized to side² x 3 to give a glance to the whole image
    normalized_patches.append(zm_mp(resize_bl(original_image, side)))
    batch_of_patches = tf.pack(normalized_patches)
    return tf.image.convert_image_dtype(
        original_image, dtype=tf.uint8), batch_of_patches


def read_and_batchify_image(image_path, shape, image_type="jpg"):
    """Return the original image as read from image_path and the image splitted as a batch tensor.
    Args:
        image_path: image path
        shape: batch shape, like: [no_patches_per_side**2, patch_side, patch_side, 3]
        image_type: image type
    Returns:
        original_image, patches
        where original image is a tensor in the format [widht, height 3]
        and patches is a tensor of processed images, ready to be classified, with size
        [batch_size, w, h, 3]"""

    original_image = read_image(image_path, 3, image_type)

    # extract values from shape
    patch_side = shape[1]
    no_patches_per_side = int(math.sqrt(shape[0]))
    resized_input_side = patch_side * no_patches_per_side

    resized_image = resize_bl(original_image, resized_input_side)

    resized_image = tf.expand_dims(resized_image, 0)
    patches = tf.space_to_depth(resized_image, patch_side)
    print(patches)
    patches = tf.squeeze(patches, [0])  #4,4,192*192*3
    print(patches)
    patches = tf.reshape(patches,
                         [no_patches_per_side**2, patch_side, patch_side, 3])
    print(patches)
    patches_a = tf.split(0, no_patches_per_side**2, patches)
    print(patches_a)
    normalized_patches = []
    for patch in patches_a:
        patch_as_input_image = zm_mp(
            tf.reshape(tf.squeeze(patch, [0]), [patch_side, patch_side, 3]))
        print(patch_as_input_image)
        normalized_patches.append(patch_as_input_image)

    # the last patch is not a "patch" but the whole image resized to patch_side² x 3
    # to give a glance to the whole image, in parallel with the patch analysis
    normalized_patches.append(zm_mp(resize_bl(original_image, patch_side)))
    batch_of_patches = tf.pack(normalized_patches)
    return tf.image.convert_image_dtype(original_image,
                                        tf.uint8), batch_of_patches
