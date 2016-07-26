#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""./heatmap_test.py --image-path <img path>"""

import argparse
import os
import sys
import time
from collections import defaultdict
import tensorflow as tf
import cv2
import numpy as np
import train
import pgnet
import pascal_input
import image_processing

# pascal sorted labels
PASCAL_LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                 "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                 "motorbike", "person", "pottedplant", "sheep", "sofa",
                 "train", "tvmonitor"]

# detection constants
PATCH_SIDE = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * 4
NO_PATCHES_PER_SIDE = 4
#eg: 768 -> 4 patch 192*192 -> each one produces a spatial map of 4x4x20 probabilities
RESIZED_INPUT_SIDE = PATCH_SIDE * NO_PATCHES_PER_SIDE

# trained pgnet constants
BACKGROUND_CLASS = 20
MIN_PROB = 0.9


def upsample_and_shift(ds_coords, downsamplig_factor, shift_amount,
                       scaling_factors):
    """Upsample ds_coords by downsampling factor, then
    shift the upsampled coordinates by shift amount, then
    resize the upsampled coordinates to the input image size, using the scaling factors.

    Args:
        ds_coords: downsampled coordinates [x0,y0, x1, y1]
        downsampling_factor: the net downsample factor, used to upsample the coordinates
        shift_amount: the quantity [2 coords] to add at each upsampled coordinate
        scaling_factors: [along_x, along_y] float numbers. Ration between net input
            and original image
    Return:
        the result of the previous described operations with shape: [x0, y0, x1, y1]
    """
    scaling_factor_x = scaling_factors[0]
    scaling_factor_y = scaling_factors[1]
    # create coordinates of rect in the downsampled image
    # convert to numpy array in order to use broadcast ops
    coord = np.array(ds_coords)

    # upsample coordinates to find the coordinates of the cell
    box = coord * downsamplig_factor

    # shift coordinates to the position of the current cell
    # in the resized input image
    box += [shift_amount[0], shift_amount[1], shift_amount[0], shift_amount[1]]

    # scale coordinates to the input image
    input_box = np.ceil(box * [scaling_factor_x, scaling_factor_y,
                               scaling_factor_x, scaling_factor_y]).astype(int)
    return input_box  # [x0, y0, x1, y1]


def main(args):
    """ main """

    if not os.path.exists(args.image_path):
        print("{} does not exists".format(args.image_path))
        return 1

    current_dir = os.path.abspath(os.getcwd())

    # Number of classes in the dataset plus 1.
    # Labelp pascal_input. NUM_CLASSES + 1 is reserved for
    # the background class.
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
        # each cell in coords (batch_position, i, j) -> is a probability vector
        per_batch_probabilities = tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes]))
        # [tested positions, num_classes]
        print(per_batch_probabilities)

        # array[0]=values, [1]=indices
        top_k = tf.nn.top_k(per_batch_probabilities, k=5)
        # each with shape [tested_positions, k]

        original_image, batch = image_processing.read_and_batchify_image(
            tf.constant(args.image_path),
            [NO_PATCHES_PER_SIDE**2, PATCH_SIDE, PATCH_SIDE, 3],
            image_type=args.image_path.split('.')[-1])

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            batchifyed_image = batch.eval()

            #for idx, img in enumerate(batchifyed_image):
            #    cv2.imshow(str(idx), img)
            #cv2.waitKey(0)
            start = time.time()
            probability_map, top_values, top_indices, image = sess.run(
                [logits, top_k[0], top_k[1], original_image],
                feed_dict={
                    "images_:0": batchifyed_image
                })
            nn_time = time.time() - start
            print("NN time: {}".format(nn_time))

            # extract image (resized image) dimensions to get the scaling factor
            # respect to the original image
            scaling_factors = np.array([image.shape[1] / RESIZED_INPUT_SIDE,
                                        image.shape[0] / RESIZED_INPUT_SIDE])

            # let's think to the net as a big net, with the last layer (before the FC
            # layers for classification) with a receptive field of
            # LAST_KERNEL_SIDE x LAST_KERNEL_SIDE. Lets approximate the net with this last kernel:
            # If the image is scaled down to  LAST_KERNEL_SIDE x LAST_KERNEL_SIDE
            # the output is a single point.
            # if the image is scaled down to something bigger
            # (that make the output side of contolution integer) the result is a spacial map
            # of points. Every point has a depth of num classes.

            # save coordinates and batch id, format: [y1, x1, y2, x2]
            # input image cooords are coords scaled up to the input image
            input_image_coords = defaultdict(list)
            # convert probability map coordinates to reshaped coordinates
            # (that contains the softmax probabilities): it's a counter.
            probability_coords = 0
            for j in range(NO_PATCHES_PER_SIDE):
                for i in range(NO_PATCHES_PER_SIDE):
                    for pmap_y in range(probability_map.shape[1]):
                        # calculate position in the downsampled image ds
                        ds_y = pmap_y * pgnet.CONV_STRIDE
                        for pmap_x in range(probability_map.shape[2]):
                            ds_x = pmap_x * pgnet.CONV_STRIDE

                            if top_values[probability_coords][
                                    0] > MIN_PROB and top_indices[
                                        probability_coords][
                                            0] != BACKGROUND_CLASS:

                                top_1_label = PASCAL_LABELS[top_indices[
                                    probability_coords][0]]

                                # create coordinates of rect in the downsampled image
                                # convert to numpy array in order to use broadcast ops
                                coord = [ds_x, ds_y,
                                         ds_x + pgnet.LAST_KERNEL_SIDE,
                                         ds_y + pgnet.LAST_KERNEL_SIDE]
                                # if something is found, append rectagle to the
                                # map of rectalges per class
                                cv_rect = upsample_and_shift(
                                    coord, pgnet.DOWNSAMPLING_FACTOR,
                                    [PATCH_SIDE * i, PATCH_SIDE * j],
                                    scaling_factors)
                                # save the probability associated to the rect
                                # [ [rect], probability]
                                input_image_coords[top_1_label].append(
                                    [cv_rect,
                                     top_values[probability_coords][0]])

                            # update probability coord value
                            probability_coords += 1

            # we processed the local regions, lets look at the global regions
            # of the whole image resized and analized
            # as the last image in the batch.
            # Here we give a glance to the image

            # probability_coords can
            # increase again by probability_map.shape[1] * probability_map.shape[2]
            # = the location watched in the original, resized, image

            # invert scaling factor, because they are the
            # ratio of the original image and the downsampled image, but i need
            # the ratio between the dowsampled image and the original image

            # new scaling factor between original image and resized image (not only to a patch)
            scaling_factors = np.array(
                [image.shape[1] / PATCH_SIDE, image.shape[0] / PATCH_SIDE])
            for pmap_y in range(probability_map.shape[1]):
                # calculate position in the downsampled image ds
                ds_y = pmap_y * pgnet.CONV_STRIDE
                for pmap_x in range(probability_map.shape[2]):
                    ds_x = pmap_x * pgnet.CONV_STRIDE

                    # TODO: merge global predictions with local predictions
                    # evaluate not only the first class (?)
                    if top_values[probability_coords][
                            0] > MIN_PROB and top_indices[probability_coords][
                                0] != BACKGROUND_CLASS:

                        top_1_label = PASCAL_LABELS[top_indices[
                            probability_coords][0]]

                        # create coordinates of rect in the downsampled image
                        # convert to numpy array in order to use broadcast ops
                        coord = [ds_x, ds_y, ds_x + pgnet.LAST_KERNEL_SIDE,
                                 ds_y + pgnet.LAST_KERNEL_SIDE]
                        # if something is found, append rectagle to the
                        # map of rectalges per class
                        cv_rect = upsample_and_shift(coord,
                                                     pgnet.DOWNSAMPLING_FACTOR,
                                                     [0, 0], scaling_factors)
                        # save the probability associated to the rect
                        # [ [rect], probability]
                        rect_prob = [cv_rect, top_values[probability_coords][0]
                                     ]
                        print('Glance: {} ({})'.format(rect_prob, top_1_label))
                        input_image_coords[top_1_label].append(rect_prob)

                    # update probability coord value
                    probability_coords += 1

            for label, rect_prob_list in input_image_coords.items():
                #rect_list, _ = cv2.groupRectangles(
                #    np.array(value[0]).tolist(), 1)

                # associate a color with a label
                rnd = lambda: np.random.randint(0, 255)
                color = [rnd(), rnd(), rnd()]

                for rect_prob in rect_prob_list:
                    rect = rect_prob[0]
                    prob = rect_prob[1]

                    cv2.rectangle(image, (rect[0], rect[1]),
                                  (rect[2], rect[3]), color, 2)
                    print(label, rect, prob)
                    cv2.putText(
                        image,
                        label, (rect[0] + 10, rect[1] + 10),
                        0,
                        1,
                        color,
                        thickness=2)

            nn_and_drawing_time = time.time() - start
            print("NN + drawing time: {}".format(nn_and_drawing_time))
            cv2.imshow("img", image)
            cv2.waitKey(0)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Apply the model to image-path")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--image-path")
    sys.exit(main(PARSER.parse_args()))
