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
import cv2
import train
import pgnet
import pascal_input
import image_processing


def read_and_resize_image(image_path, output_side, image_type="jpg"):
    side = 250
    """Read the entire image and resize it to 448x448"""
    #TODO: use image_processing.eval_image to parse each box
    if image_type == "jpg":
        image = image_processing.read_image_jpg(image_path, 3)
    else:
        image = image_processing.read_image_png(image_path, 3)

    image = image_processing.resize_bl(image, side)
    return image


def main(args):
    """ main """

    if not os.path.exists(args.image_path):
        print("{} does not exists".format(args.image_path))
        return 1

    current_dir = os.path.abspath(os.getcwd())

    # Number of classes in the dataset plus 1.
    # Labelp pascal_input. NUM_CLASSES + 1 is reserved for
    # an (unused) background class.
    num_classes = pascal_input.NUM_CLASSES + 1

    # export model.pb from session dir. Skip if model.pb already exists
    pgnet.export_model(num_classes, current_dir + "/session", "model-0",
                       "model.pb")

    with tf.Graph().as_default() as graph, tf.device(args.device):
        const_graph_def = tf.GraphDef()
        with open(train.TRAINED_MODEL_FILENAME, 'rb') as saved_graph:
            const_graph_def.ParseFromString(saved_graph.read())
            # replace current graph with the saved graph def (and content)
            # name="" is importat because otherwise (with name=None)
            # the graph definitions will be prefixed with import.
            tf.import_graph_def(const_graph_def, name="")

        # now the current graph contains the trained model

        # (?, n, n, NUM_CLASSES) tensor
        logits = graph.get_tensor_by_name(pgnet.OUTPUT_TENSOR_NAME + ":0")

        # array[0]=values, [1]=indices
        top_k = tf.nn.top_k(logits, k=5)

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            image = tf.expand_dims(
                read_and_resize_image(
                    tf.constant(args.image_path),
                    pgnet.INPUT_SIDE,
                    image_type=args.image_path.split('.')[-1]),
                0).eval()

            cv2.imshow("resized", image[0])
            cv2.waitKey(0)

            predictions_prob, top_values, top_indices = sess.run(
                [logits, top_k[0], top_k[1]],
                feed_dict={
                    "images_:0": image,
                })

            # remove batch size (we're processing one image at time)
            predictions_prob = predictions_prob[0]
            print(predictions_prob)
            print(top_values)
            print(top_indices)
            print("Predictions: ", predictions_prob.size,
                  predictions_prob.shape)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Apply the model to image-path")
    PARSER.add_argument("--device", default="/gpu:0")
    PARSER.add_argument("--image-path")
    sys.exit(main(PARSER.parse_args()))
