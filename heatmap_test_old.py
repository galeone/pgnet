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
import math
from collections import defaultdict
from statistics import mode, StatisticsError
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
PATCH_SIDE = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * 10
NO_PATCHES_PER_SIDE = 4
#eg: 768 -> 4 patch 192*192 -> each one produces a spatial map of 4x4x20 probabilities
RESIZED_INPUT_SIDE = PATCH_SIDE * NO_PATCHES_PER_SIDE
MIN_GLOBAL_PROB = 0.95
MIN_LOCAL_PROB = 0.95
MIN_GLOCAL_PROB = 0.6
TOP_K = 2

# trained pgnet constants
BACKGROUND_CLASS = 20


def rnd_color():
    """ Generate random colors in RGB format"""
    rnd = lambda: np.random.randint(0, 255)
    return (rnd(), rnd(), rnd())


LABEL_COLORS = {label: rnd_color() for label in PASCAL_LABELS}


def legend():
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    y = 20
    for label in PASCAL_LABELS:
        color = LABEL_COLORS[label]
        cv2.putText(image, label, (5, y), 0, 1, color, thickness=2)
        y += 20

    cv2.imshow("legend", image)


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


def draw_box(image, rect, label, color, thickness=1):
    """ Draw rect on image, writing label into rect and colors border and text with color"""
    cv2.rectangle(
        image, (rect[0], rect[1]), (rect[2], rect[3]),
        color,
        thickness=thickness)
    cv2.putText(
        image,
        label, (rect[0] + 10, rect[1] + 10),
        0,
        1,
        color,
        thickness=thickness)


def intersect(rect_a, rect_b):
    """Returns true if rect_a intersects rect_b"""
    x = max(rect_a[0], rect_b[0])
    y = max(rect_a[1], rect_b[1])
    w = min(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2]) - x
    h = min(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3]) - y
    if w < 0 or h < 0:
        return False
    return True


def contains(rect_a, rect_b):
    """Returns true if rect_a contains rect_b"""
    return rect_a[0] <= rect_b[0] and rect_a[1] <= rect_b[1] and rect_a[
        2] >= rect_b[2] and rect_a[3] >= rect_b[3]


def center_point(rect):
    """Extract the coordinates (rounted to int) of the center of the rect"""
    h = rect[2] - rect[0]
    w = rect[3] - rect[1]
    return (2 * rect[0] + h, 2 * rect[1] + w)


def norm(p0):
    return math.sqrt(p0[0]**2 + p0[1]**2)


def l2(p0, p1):
    return norm((p0[0] - p1[0], p0[1] - p1[1]))


def group_overlapping_with_same_class(map_of_regions):
    """merge overlapping rectangles with the same class
    Merge only if there's overlapping between at leat 2 regions.
    Args:
        map_of_regions:  {"label": [[rect1, p1, rank], [rect2, p2, rank], ..], "label2"...}
    """
    grouped_map = defaultdict(list)
    for label, rect_prob_list in map_of_regions.items():
        # extract rectangles from the array of pairs
        rects_only = np.array([value[0] for value in rect_prob_list])
        # group them
        rect_list, _ = cv2.groupRectangles(rects_only.tolist(), 1, eps=0.5)
        # calculate probability of the grouped rectangles as the mean prob
        merged_rect_prob_rank_list = []
        for merged_rect in rect_list:
            sum_of_prob = 0.0
            sum_of_rank = 0.0
            merged_count = 0
            for idx, original_rect in enumerate(rects_only):
                if intersect(original_rect, merged_rect):
                    original_rect_prob = rect_prob_list[idx][1]
                    original_rect_rank = rect_prob_list[idx][2]
                    sum_of_prob += original_rect_prob
                    sum_of_rank += original_rect_rank
                    merged_count += 1

            avg_prob = sum_of_prob / merged_count
            avg_rank = sum_of_rank / merged_count
            merged_rect_prob_rank_list.append(
                (merged_rect, avg_prob, avg_rank))

        if len(rect_list) > 0:
            grouped_map[label] = merged_rect_prob_rank_list
    return grouped_map


def draw_final(image, boxes):
    #unique = {}
    for box, label_prob_lists in boxes.items():
        top_1_label_prob = ['', 0.0]
        for label, rect_prob_list in label_prob_lists.items():
            for rect_prob in rect_prob_list:
                prob = rect_prob[1]
                if prob > top_1_label_prob[1]:
                    top_1_label_prob = [label, prob]
        #unique[box] = top_1_label_prob
        draw_box(image, box, "{} {:.3}".format(top_1_label_prob[0],
                                               top_1_label_prob[1]),
                 LABEL_COLORS[top_1_label_prob[0]])


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

        # array[0]=values, [1]=indices
        top_k = tf.nn.top_k(per_batch_probabilities, k=TOP_K)
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
            patch_scaling_factors = np.array(
                [image.shape[1] / RESIZED_INPUT_SIDE,
                 image.shape[0] / RESIZED_INPUT_SIDE])
            # scaling factor between original image and resized image
            full_image_scaling_factors = np.array(
                [image.shape[1] / PATCH_SIDE, image.shape[0] / PATCH_SIDE])

            # let's think to the net as a big net, with the last layer (before the FC
            # layers for classification) with a receptive field of
            # LAST_KERNEL_SIDE x LAST_KERNEL_SIDE. Lets approximate the net with this last kernel:
            # If the image is scaled down to  LAST_KERNEL_SIDE x LAST_KERNEL_SIDE
            # the output is a single point.
            # if the image is scaled down to something bigger
            # (that make the output side of contolution integer) the result is a spacial map
            # of points. Every point has a depth of num classes.

            # convert probability map coordinates to reshaped coordinates
            # (that contains the softmax probabilities): it's a counter.
            probability_coords = 0

            # boxes is the image grid
            boxes = defaultdict(lambda: defaultdict(list))

            for j in range(NO_PATCHES_PER_SIDE):
                for i in range(NO_PATCHES_PER_SIDE):
                    # grid coordinates in the original image
                    corner_top_x = int(i * PATCH_SIDE *
                                       patch_scaling_factors[0])
                    corner_top_y = int(j * PATCH_SIDE *
                                       patch_scaling_factors[1])
                    corner_bottom_x = int(
                        (i + 1) * PATCH_SIDE * patch_scaling_factors[0])
                    corner_bottom_y = int(
                        (j + 1) * PATCH_SIDE * patch_scaling_factors[1])
                    grid_coord = (corner_top_x, corner_top_y, corner_bottom_x,
                                  corner_bottom_y)

                    for pmap_y in range(probability_map.shape[1]):
                        # calculate position in the downsampled image ds
                        ds_y = pmap_y * pgnet.CONV_STRIDE
                        for pmap_x in range(probability_map.shape[2]):
                            ds_x = pmap_x * pgnet.CONV_STRIDE

                            # if is not background and the has the right prob
                            if top_indices[probability_coords][
                                    0] != BACKGROUND_CLASS and top_values[
                                        probability_coords][
                                            0] > MIN_LOCAL_PROB:

                                # create coordinates of rect in the downsampled image
                                ds_coord = np.array(
                                    [ds_x, ds_y, ds_x + pgnet.LAST_KERNEL_SIDE,
                                     ds_y + pgnet.LAST_KERNEL_SIDE])

                                # get the input coordinates
                                rect = upsample_and_shift(
                                    ds_coord, pgnet.DOWNSAMPLING_FACTOR,
                                    [PATCH_SIDE * i, PATCH_SIDE * j],
                                    patch_scaling_factors)

                                for top_k in range(TOP_K):
                                    prob = top_values[probability_coords][
                                        top_k]
                                    rect_prob_rank = [rect, prob, top_k + 1]
                                    label = PASCAL_LABELS[top_indices[
                                        probability_coords][top_k]]
                                    boxes[grid_coord][label].append(
                                        rect_prob_rank)

                                # update probability coord value
                            probability_coords += 1

            # we processed the local regions, lets look at the global regions
            # of the whole image resized and analized
            # as the last image in the batch.
            # Here we give a glance to the image

            # probability_coords can
            # increase again by probability_map.shape[1] * probability_map.shape[2]
            # = the location watched in the original, resized, image

            # save the global glance in a separate dict
            global_glance = defaultdict(list)
            for pmap_y in range(probability_map.shape[1]):
                # calculate position in the downsampled image ds
                ds_y = pmap_y * pgnet.CONV_STRIDE
                for pmap_x in range(probability_map.shape[2]):
                    ds_x = pmap_x * pgnet.CONV_STRIDE

                    if top_indices[probability_coords][
                            0] != BACKGROUND_CLASS and top_values[
                                probability_coords][0] > MIN_GLOBAL_PROB:

                        # create coordinates of rect in the downsampled image
                        # convert to numpy array in order to use broadcast ops
                        coord = [ds_x, ds_y, ds_x + pgnet.LAST_KERNEL_SIDE,
                                 ds_y + pgnet.LAST_KERNEL_SIDE]
                        # if something is found, append rectagle to the
                        # map of rectalges per class
                        cv_rect = upsample_and_shift(
                            coord, pgnet.DOWNSAMPLING_FACTOR, [0, 0],
                            full_image_scaling_factors)

                        top_1_label = PASCAL_LABELS[top_indices[
                            probability_coords][0]]
                        # save the probability associated to the rect
                        # [ [rect], probability]
                        rect_prob_rank = [cv_rect,
                                          top_values[probability_coords][0], 1]
                        print('Glance: {} ({})'.format(rect_prob_rank,
                                                       top_1_label))
                        global_glance[top_1_label].append(rect_prob_rank)

                    # update probability coord value
                    probability_coords += 1

            # global
            global_rect_prob = group_overlapping_with_same_class(global_glance)
            # local, for every cell in the grid
            unique_boxes = defaultdict(list)
            for box_coord, local_glance in boxes.items():
                unique_boxes[box_coord] = group_overlapping_with_same_class(
                    local_glance)
            #unique_boxes = boxes

            draw_local = False
            draw_global = True
            if draw_global:
                for global_label, global_rect_prob_list in global_rect_prob.items(
                ):
                    for rect_prob in global_rect_prob_list:
                        rect = rect_prob[0]
                        prob = rect_prob[1]
                        draw_box(
                            image,
                            rect,
                            "{} {:.3}".format(global_label, prob),
                            LABEL_COLORS[global_label],
                            thickness=1)

            if draw_local:
                for box_coord, local_glance in unique_boxes.items():
                    for local_label, local_rect_prob_rank_list in local_glance.items(
                    ):
                        for rect_prob in local_rect_prob_rank_list:
                            rect = rect_prob[0]
                            prob = rect_prob[1]
                            rank = rect_prob[2]
                            draw_box(image, rect, "({}){} {:.3}".format(
                                rank, local_label, prob),
                                     LABEL_COLORS[local_label])

            # global regions are research area for local region
            # search within top-k local regions if there's the global region
            # label. If it's, promote it, assigning it the global regions probability
            # Do this for every intersection.
            # eg: if local region A, is covered by 2 global regions (with different class)
            # and in the top-k labels associated with A there are the 2 labels of the global regions
            # promote the local labels (assign to them the global region probability).
            # Than, draw the local label, labeled with the highest probability.
            intersections = defaultdict(lambda: defaultdict(list))

            for global_label, global_rect_prob_list in global_rect_prob.items(
            ):
                for g_rect_prob in global_rect_prob_list:
                    g_rect = g_rect_prob[0]
                    g_prob = g_rect_prob[1]

                    g_center = np.array(center_point(g_rect))
                    normalized_g_center = g_center / norm(g_center)

                    for box_coord, local_glance in unique_boxes.items():
                        # fast filter
                        if intersect(box_coord, g_rect):
                            for local_label, local_rect_prob_rank_list in local_glance.items(
                            ):
                                if global_label == local_label:
                                    for idx, l_rect_prob_rank in enumerate(
                                            local_rect_prob_rank_list):
                                        l_rect = l_rect_prob_rank[0]
                                        l_prob = l_rect_prob_rank[1]
                                        l_rank = l_rect_prob_rank[2]
                                        if intersect(g_rect, l_rect):
                                            # promote l_rect, based on the rank (in the top k)
                                            # of l_rect
                                            promotion = max(g_prob, l_prob)

                                            if promotion < MIN_GLOCAL_PROB:
                                                promotion = 0

                                            pair = (l_rect, promotion)
                                            print(
                                                'Update {} to {}. [global {}, local: {}]'.
                                                format(unique_boxes[box_coord][
                                                    local_label][idx], pair,
                                                       g_rect, l_rect))

                                            # update probability
                                            unique_boxes[box_coord][
                                                local_label][idx] = pair
                                            intersections[box_coord][
                                                local_label].append(pair)

                                            # and penalize classes, for the same location, without global intersection

                                        # draw top-1 only of adjusted local glances
            for box_coord, local_glance in intersections.items():
                for local_label, local_rect_prob_list in local_glance.items():
                    print('cell: {}, classes: {} {}, {}'.format(
                        box_coord, local_label, local_rect_prob_list, len(
                            local_rect_prob_list)))
                    for rect_prob in local_rect_prob_list:
                        rect = rect_prob[0]
                        prob = rect_prob[1]
                        if prob > 0:
                            draw_box(
                                image,
                                rect,
                                "{} {:.3}".format(local_label, prob),
                                LABEL_COLORS[local_label],
                                thickness=2)
                        else:
                            print('*')

            #print(intersections)
            #draw_final(image, intersections)
            """
            # if the global glances resulted in one single class
            # there's an high probability that the image contains 1 element in foreground
            # so, discard local regions and use only the detected global regions
            num_glance_classes = len(global_rect_prob)
            if num_glance_classes == 1:
                for label, rect_prob_list in global_rect_prob.items():
                    # extract rectangles from the array of pairs
                    # rects are already grouped
                    for rect_prob in rect_prob_list:
                        rect = rect_prob[0]
                        prob = rect_prob[1]
                        print(rect, label, prob)
                        draw_box(image, rect, label + " {:.3}".format(prob),
                                 LABEL_COLORS[label])

            """
            nn_and_drawing_time = time.time() - start
            print("NN + drawing time: {}".format(nn_and_drawing_time))
            cv2.imshow("img", image)
            legend()
            cv2.waitKey(0)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Apply the model to image-path")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--image-path")
    sys.exit(main(PARSER.parse_args()))
