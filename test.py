import os
import cv2
import tensorflow as tf
import numpy as np
import pgnet

image_path = "/data/PASCAL_2012_cropped/2010_003078.jpg"
image = cv2.imread(image_path)

# the placeholder definition is important
images_ = tf.placeholder(tf.float32, [None, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE,
                                      pgnet.INPUT_DEPTH])
model = pgnet.get(images_, keep_prob=1, num_class=20)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    reshaped_images = tf.image.resize_nearest_neighbor(
        tf.reshape(image, [-1, image.shape[0], image.shape[1], image.shape[2]
                           ]).eval(),
        [pgnet.INPUT_SIDE, pgnet.INPUT_SIDE]).eval()

    out = sess.run(model, feed_dict={images_: reshaped_images})

    print(out)
