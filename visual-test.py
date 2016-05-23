import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import pgnet

image_path = "/data/PASCAL_2012_cropped/2010_003078.jpg"
image = cv2.imread(image_path)
image1 = cv2.imread(image_path)

images_ = tf.placeholder(tf.float32, [None, image.shape[0], image.shape[1], 3])

#model = pgnet.get(images_)

kernels = tf.get_variable("kernels", [3, 3, 3, 3],
                          initializer=tf.random_normal_initializer())

reshaped_input_tensor = tf.reshape(
    tf.pack([tf.cast(image, tf.float32), tf.cast(image1, tf.float32)]),
    [-1, image.shape[0], image.shape[1], image.shape[2]])
print(reshaped_input_tensor)

eq_op = pgnet.eq_conv(reshaped_input_tensor, 3, 3, 2)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(image.shape)
    print(kernels.get_shape())
    reshaped_input = reshaped_input_tensor.eval()
    cv2.imshow("original", tf.cast(reshaped_input[0], tf.uint8).eval())
    """
    out = sess.run(tf.nn.relu(tf.nn.conv2d(reshaped_input, kernels,
                                           [1, 1, 1, 1], 'VALID').eval()))

    print("Conv2d")
    print(out.shape)
    print(tf.shape(out))


    cv2.imshow("im",
               cv2.normalize(out[0],
                             out[0],
                             alpha=0,
                             beta=2**24,
                             norm_type=cv2.NORM_MINMAX))
    #cv2.imshow("im",out[0])

    out_tensor = tf.nn.relu(tf.nn.atrous_conv2d(
        reshaped_input, kernels, 20,
        padding='VALID'))


    out = out_tensor.eval()
    print("Atrous conv2d")
    print(out.shape)
    """
    """cv2.imshow("im1",
               cv2.normalize(out[0],
                             out[0],
                             alpha=0,
                             beta=2**24,
                             norm_type=cv2.NORM_MINMAX))"""
    #cv2.imshow("atrous", out[0])

    #padded_tensor = padder(reshaped_input_tensor, out_tensor)
    #padded = padded_tensor.eval()
    #print(padded)

    #cv2.imshow("impadded", cv2.normalize(padded[0], padded[0], alpha=0, beta=2**24, norm_type=cv2.NORM_MINMAX))
    #cv2.imshow("imp", tf.cast(padded[0], tf.uint8).eval())

    #always remember to cast to tf.uint8 if you want to display the result with opencv
    eq_out = sess.run(eq_op)
    cv2.imshow("eq conv", tf.cast(eq_out[0], tf.uint8).eval())

    cv2.waitKey(0)
