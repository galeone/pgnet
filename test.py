import os
import cv2
import tensorflow as tf
import numpy as np
import pgnet

image_path = "/data/PASCAL_2012_cropped/2010_003078.jpg"
image = cv2.imread(image_path)
images_ = tf.placeholder(tf.float32, [None, image.shape[0], image.shape[1], 3])
model = pgnet.get(images_)

#kernels = tf.get_variable("kernels", [3, 3, 3, 3],
#                         initializer=tf.random_normal_initializer())
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    out = sess.run(
        model,
        feed_dict={
            images_: tf.reshape(
                tf.cast(image, tf.float32).eval(),
                [-1, image.shape[0], image.shape[1], image.shape[2]]).eval()
        })

    print(model)
    print(out)

    #cv2.imshow("impadded", cv2.normalize(padded[0], padded[0], alpha=0, beta=2**24, norm_type=cv2.NORM_MINMAX))
    cv2.imshow("imp", tf.cast(out[0], tf.uint8).eval())

    #always rememer to cast to tf.uint8 if you want to display the result with opencv
    #cv2.imshow("original", tf.cast(reshaped_input[0], tf.uint8).eval())

    cv2.waitKey(0)
