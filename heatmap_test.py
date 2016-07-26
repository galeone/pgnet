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


def rnd_color():
    """ Generate random colors in RGB format"""
    rnd = lambda: np.random.randint(0, 255)
    return [rnd(), rnd(), rnd()]


LABEL_COLORS = {label: rnd_color() for label in PASCAL_LABELS}

# detection constants
PATCH_SIDE = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * 4
NO_PATCHES_PER_SIDE = 4
#eg: 768 -> 4 patch 192*192 -> each one produces a spatial map of 4x4x20 probabilities
RESIZED_INPUT_SIDE = PATCH_SIDE * NO_PATCHES_PER_SIDE

# trained pgnet constants
BACKGROUND_CLASS = 20
MIN_GLOBAL_PROB = 0.9
MIN_LOCAL_PROB = 0.8


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
        the result of the previous described operations as a tuple: (x0, y0, x1, y1)
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
    return tuple(input_box)  # (x0, y0, x1, y1)


def draw_box(image, rect, label, color):
    """ Draw rect on image, writing label into rect and colors border and text with color"""
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)
    cv2.putText(
        image, label, (rect[0] + 20, rect[1] + 20), 0, 1, color, thickness=2)


def intersect(a, b):
    """Returns true of a intersects b"""
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return Flase
    return True


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
        top_k = tf.nn.top_k(per_batch_probabilities, k=3)
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

            # for every region in the original image, save its coordinates
            # and store the top-k labels associated
            # the following default dict can be used with tuples:
            # (x0,y0,x1,y1) = [list, of, elements]
            batch_to_input_image_coords = defaultdict(int)
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

                            # if is not background and the has the right prob
                            if top_values[probability_coords][
                                    0] > MIN_LOCAL_PROB and top_indices[
                                        probability_coords][
                                            0] != BACKGROUND_CLASS:

                                # create coordinates of rect in the downsampled image
                                coord = np.array(
                                    [ds_x, ds_y, ds_x + pgnet.LAST_KERNEL_SIDE,
                                     ds_y + pgnet.LAST_KERNEL_SIDE])

                                # get the input coordinates
                                input_box = upsample_and_shift(
                                    coord, pgnet.DOWNSAMPLING_FACTOR,
                                    [PATCH_SIDE * i, PATCH_SIDE * j],
                                    scaling_factors)

                                # save top{1,2,3} label for the current region. Format: x[(coords)] = [label, prob]
                                batch_to_input_image_coords[input_box] = [
                                    [
                                        PASCAL_LABELS[top_indices[
                                            probability_coords][top_k]],
                                        top_values[probability_coords][top_k]
                                    ] for top_k in range(3)
                                ]

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
            # save the global glance in a separate dict
            resized_to_input_image_coords = defaultdict(list)
            for pmap_y in range(probability_map.shape[1]):
                # calculate position in the downsampled image ds
                ds_y = pmap_y * pgnet.CONV_STRIDE
                for pmap_x in range(probability_map.shape[2]):
                    ds_x = pmap_x * pgnet.CONV_STRIDE

                    if top_values[probability_coords][
                            0] > MIN_GLOBAL_PROB and top_indices[
                                probability_coords][0] != BACKGROUND_CLASS:

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
                        rect_prob = [cv_rect,
                                     top_values[probability_coords][0]]
                        print('Glance: {} ({})'.format(rect_prob, top_1_label))
                        resized_to_input_image_coords[top_1_label].append(
                            rect_prob)

                    # update probability coord value
                    probability_coords += 1

            # if the global glances resulted in one single class
            # there's an high probability that the image contains 1 element in forground
            # so, discard local regions and use only the detected global regions
            # TODO: merge overlapping rectangles
            num_glance_classes = len(resized_to_input_image_coords)
            if num_glance_classes == 1:
                print('Final labels')
                for label, rect_prob_list in resized_to_input_image_coords.items(
                ):
                    # extract rectangles from the array of pairs
                    rects_only = [value[0] for value in rect_prob_list]
                    avg_prob = sum(
                        [value[1]
                         for value in rect_prob_list]) / len(rect_prob_list)
                    rect_list, _ = cv2.groupRectangles(
                        np.array(rects_only).tolist(), 1)
                    for rect in rect_list:
                        print(rect, label, avg_prob)
                        draw_box(image, rect, label + " " + str(avg_prob),
                                 LABEL_COLORS[label])
            else:
                # exploit global glance to increase the probability of local regions
                # that are under a global region, with the same class.
                # Let's look not only at the top-1 label, but down to the top-3.
                # If in the top-3 label of the local regions, we find the global label
                # that's convering the regions, replace the top label with the global label.
                # Than, create a region of adiacent boxes
                for global_label, global_rect_prob_list in resized_to_input_image_coords.items(
                ):
                    # extract rectangles from the array of pairs
                    rects_only = [value[0] for value in global_rect_prob_list]
                    avg_global_prob = sum(
                        [value[1] for value in global_rect_prob_list]) / len(
                            global_rect_prob_list)
                    global_rect_list, _ = cv2.groupRectangles(
                        np.array(rects_only).tolist(), 1)

                    # global rect has the coordinates of the global rectangle
                    for global_rect in global_rect_list:
                        for local_rect, local_labels_prob in batch_to_input_image_coords.items(
                        ):
                            # if there's intersection among global and local rect
                            if intersect(global_rect, local_rect):
                                # if the global label is in the top-k labels of the regions
                                local_labels_only = [
                                    value[0] for value in local_labels_prob
                                ]
                                print('looking for {} in {}'.format(
                                    global_label, local_labels_only))
                                if global_label in local_labels_only:
                                    # set the top label to be the global label
                                    # 0 -> top-1 pair
                                    # [ 0 = label, 1 prob]
                                    avg_prob = (batch_to_input_image_coords[
                                        local_rect][0][1] + avg_global_prob
                                                ) / 2
                                    new_top_1_label_prob = [global_label,
                                                            avg_prob]
                                    print('replaced (top-1) {} with {}'.format(
                                        batch_to_input_image_coords[
                                            local_rect][
                                                0], new_top_1_label_prob))
                                    batch_to_input_image_coords[local_rect][
                                        0] = new_top_1_label_prob

                # draw top-1 only, merge elements by class
                glocal_regions = defaultdict(list)
                for local_rect, local_labels_prob in batch_to_input_image_coords.items(
                ):
                    top_1_label = local_labels_prob[0][0]
                    top_1_prob = local_labels_prob[0][1]
                    if top_1_prob > MIN_GLOBAL_PROB:
                        rect_prob = [local_rect, top_1_prob]
                        glocal_regions[top_1_label].append(rect_prob)
                        print(local_rect, top_1_label, top_1_prob)

                # merge overlapping rectangles with same label
                for label, rect_prob_list in glocal_regions.items():
                    # extract rectangles from the array of pairs
                    rects_only = [value[0] for value in rect_prob_list]
                    avg_prob = sum(
                        [value[1]
                         for value in rect_prob_list]) / len(rect_prob_list)
                    rect_list, _ = cv2.groupRectangles(
                        np.array(rects_only).tolist(), 1)
                    for rect in rect_list:
                        draw_box(image, rect, label + " " + str(avg_prob),
                                 LABEL_COLORS[label])

                        print(rect, label, avg_prob)

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
