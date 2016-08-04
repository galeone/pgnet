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


def rnd_color():
    """ Generate random colors in RGB format"""
    rnd = lambda: np.random.randint(0, 255)
    return (rnd(), rnd(), rnd())


LABEL_COLORS = {label: rnd_color() for label in PASCAL_LABELS}

# detection constants
PATCH_SIDE = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * 2
NO_PATCHES_PER_SIDE = 4
#eg: 768 -> 4 patch 192*192 -> each one produces a spatial map of 4x4x20 probabilities
RESIZED_INPUT_SIDE = PATCH_SIDE * NO_PATCHES_PER_SIDE
MIN_GLOBAL_PROB = 0.95
MIN_LOCAL_PROB = MIN_GLOBAL_PROB
TOP_K = 5

# trained pgnet constants
BACKGROUND_CLASS = 20


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
    cv2.rectangle(
        image, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness=1)
    cv2.putText(
        image, label, (rect[0] + 20, rect[1] + 20), 0, 1, color, thickness=1)


def intersect(rect_a, rect_b):
    """Returns true if rect_a intersects rect_b"""
    x = max(rect_a[0], rect_b[0])
    y = max(rect_a[1], rect_b[1])
    w = min(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2]) - x
    h = min(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3]) - y
    if w < 0 or h < 0:
        return False
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

            # for every region in the original image, save its coordinates
            # and store the top-k labels associated
            # the following default dict can be used with tuples:
            # (x0,y0,x1,y1) = [list, of, elements]
            batch_to_input_image_coords = defaultdict(int)
            # convert probability map coordinates to reshaped coordinates
            # (that contains the softmax probabilities): it's a counter.
            probability_coords = 0

            # boxes is the image grid
            boxes = defaultdict(list)

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
                                    patch_scaling_factors)

                                # save top{1, ..., TOP_K} label for the current region.
                                # Format: x[(coords)] = (label, prob)
                                batch_to_input_image_coords[input_box] = [
                                    [
                                        PASCAL_LABELS[top_indices[
                                            probability_coords][top_k]],
                                        top_values[probability_coords][top_k]
                                    ] for top_k in range(TOP_K)
                                ]

                                boxes[grid_coord].append(
                                    batch_to_input_image_coords[input_box])

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
                        cv_rect = upsample_and_shift(
                            coord, pgnet.DOWNSAMPLING_FACTOR, [0, 0],
                            full_image_scaling_factors)
                        # save the probability associated to the rect
                        # [ [rect], probability]
                        rect_prob = [cv_rect,
                                     top_values[probability_coords][0]]
                        print('Glance: {} ({})'.format(rect_prob, top_1_label))
                        resized_to_input_image_coords[top_1_label].append(
                            rect_prob)

                    # update probability coord value
                    probability_coords += 1

            # merge global overlapping rectangles with the same class.
            # merge only if there's overlapping between at leat 2 regions
            global_rect_prob = defaultdict(list)
            for label, rect_prob_list in resized_to_input_image_coords.items():
                # extract rectangles from the array of pairs
                rects_only = np.array([value[0] for value in rect_prob_list])
                # group them
                rect_list, _ = cv2.groupRectangles(rects_only.tolist(), 1)
                # calculate probability of the grouped rectangles as the mean prob
                merged_rect_prob_list = []
                for merged_rect in rect_list:
                    sum_of_prob = 0.0
                    merged_count = 0
                    for idx, original_rect in enumerate(rects_only):
                        if intersect(original_rect, merged_rect):
                            original_rect_prob = rect_prob_list[idx][1]
                            sum_of_prob += original_rect_prob
                            merged_count += 1
                    merged_rect_prob_list.append(
                        (merged_rect, sum_of_prob / merged_count))

                if len(rect_list) > 0:
                    global_rect_prob[label] = merged_rect_prob_list

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
            else:
                # exploit global glance to increase the probability of local regions
                # that are under a global region, with the same class.
                # Let's look not only at the top-1 label, but down to the TOP_K.
                # If in the TOP_K label of the local regions, we find the global label
                # that's convering the regions, replace the top label with the global label.
                # Than, create a region of adiacent boxes
                replacement_queue = defaultdict(
                    list)  # (x0,y0,x1,y1) -> (local_rect_coords) = [(class, prob)]
                for global_label, global_rect_prob_list in global_rect_prob.items(
                ):
                    # look for intersection of global rect and local rect
                    # if the intersection exists, check if the local rect has in its top-k class
                    # the current global_rect class.
                    # Create a queue of substitutions: add the class and the global prob in
                    # the queue assigned to the current local rect.
                    # In the end, evaluate the queue associated to each local rect and assign to
                    # the local rect the global class with the highest probability
                    for global_rect_prob in global_rect_prob_list:
                        global_rect = global_rect_prob[0]
                        global_prob = global_rect_prob[1]

                        for local_rect, local_labels_prob in batch_to_input_image_coords.items(
                        ):
                            # if there's intersection among global and local rect
                            if intersect(global_rect, local_rect):
                                for idx, value in enumerate(local_labels_prob):
                                    local_label = value[0]
                                    local_prob = value[1]
                                    if global_label == local_label:
                                        print(
                                            '{} intersect {} found {}. glob {} vs {}'.format(
                                                global_rect, local_rect,
                                                local_label, global_prob,
                                                local_prob))
                                        # normalize difference of probabilities between 0 and 1
                                        # use idx+1 to scale the difference
                                        normalized = 1 - (
                                            abs(global_prob - local_prob)) / (
                                                idx + 1)
                                        print(
                                            'normalized difference: {}'.format(
                                                normalized))
                                        replacement_queue[local_rect].append(
                                            (global_label, normalized))
                                        break

                # do replacements
                print('Replacements...')
                for rect, label_prob_list in replacement_queue.items():
                    # looking for the global label with the highest probability
                    # among the ones in the label_prob_list

                    max_label_prob = ['', 0]
                    for label_prob in label_prob_list:
                        if label_prob[1] > max_label_prob[1]:
                            max_label_prob = label_prob

                    # replace the probability associated with the max_label in the current rect,
                    # if the probability is lower than the max-one in the replacemente queue
                    for idx, label_prob in enumerate(
                            batch_to_input_image_coords[rect]):
                        if label_prob[0] == max_label_prob[
                                0] and max_label_prob[1] > label_prob[1]:
                            print('{}: replacing {} with {}'.format(
                                rect, batch_to_input_image_coords[rect][
                                    idx], max_label_prob))
                            batch_to_input_image_coords[rect][
                                idx] = max_label_prob
                            break

                for box, label_prob_lists in boxes.items():
                    unique = defaultdict(lambda: defaultdict(int))
                    for label_prob_list in label_prob_lists:
                        for label_prob in label_prob_list:
                            label = label_prob[0]
                            prob = label_prob[1]
                            unique[label]["sum"] += prob
                            unique[label]["count"] += 1

                    top_1_label_prob = ['', 0.0]
                    for label, dicts in unique.items():
                        unique[label]["avg"] = unique[label]["sum"] / unique[
                            label]["count"]
                        if unique[label]["avg"] > top_1_label_prob[1]:
                            top_1_label_prob = [label, unique[label]["avg"]]

                    print(unique)

                    print('box', box, 'label: ', top_1_label_prob[0], 'prob: ',
                          top_1_label_prob[1])
                    #if top_1_label_prob[1] > MIN_GLOCAL_PROB:
                    draw_box(image, box, "{} {:.3}".format(
                        top_1_label_prob[0], top_1_label_prob[1]),
                             LABEL_COLORS[top_1_label_prob[0]])
                    #else:
                    #    print('^ skipped')

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
