#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""./test.py --image-path <img path>"""

import argparse
import os
import sys
import tensorflow as tf
import pgnet
import train


def main(args):
    """ main """

    if not os.path.exists(args.image_path):
        print("{} does not exists".format(args.image_path))
        return 1

    # export model.pb from session dir. Skip if model.pb already exists
    train.export_model()

    with tf.Graph().as_default() as graph, tf.device(args.device):
        const_graph_def = tf.GraphDef()
        with open(train.TRAINED_MODEL_FILENAME, 'rb') as saved_graph:
            const_graph_def.ParseFromString(saved_graph.read())
            # replace current graph with the saved graph def (and content)
            # name="" is importat because otherwise (with name=None)
            # the graph definitions will be prefixed with import.
            tf.import_graph_def(const_graph_def, name="")

        # now the current graph contains the trained model

        # exteact the pgnet output from the graph and scale the result
        # using softmax

        softmax_linear = graph.get_tensor_by_name("softmax_linear/out:0")
        # softmax_linear is the output of a 1x1xNUM_CLASS convolution
        # to use the softmax we have to reshape it back to (?,NUM_CLASS)
        #softmax_linear = tf.reshape(softmax_linear, [-1, pgnet.NUM_CLASS])
        #softmax = tf.nn.softmax(softmax_linear, name="softmax")

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            # prepend the path
            image_path = tf.constant(args.image_path)
            # read the image
            image = tf.image.decode_jpeg(tf.read_file(image_path))

            # subtract off the mean and divide by the variance of the pixels
            image = tf.image.per_image_whitening(image)

            # pgnet accepts a batch of images as input, add the "batch" dimension.
            image = tf.expand_dims(image, 0)

            # feed the input placeholder _images with the current "batch" (1) of image
            image_evaluated = image.eval()
            predictions_prob = sess.run(softmax_linear,
                                        feed_dict={
                                            "keep_prob_:0": 1.0,
                                            "images_:0": image_evaluated,
                                        })

            # remove batch size (we're processing one image at time)
            predictions_prob = predictions_prob[0]
            print(predictions_prob)
            print(predictions_prob.size)
            print(predictions_prob.shape)


if __name__ == "__main__":
    # pylint: disable=C0103
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--device", default="/gpu:1")
    parser.add_argument("--image-path")
    sys.exit(main(parser.parse_args()))
