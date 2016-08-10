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
import operator
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
INPUT_SIDE = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * 20
OUTPUT_SIDE = INPUT_SIDE / pgnet.DOWNSAMPLING_FACTOR - pgnet.LAST_KERNEL_SIDE + 1
LOOKED_POS = OUTPUT_SIDE**2
MIN_PROB = 0.6
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


def variance(set_of_values):
    """Returns the mean and the variance of the set_of_values set"""
    n = len(set_of_values)
    sum_of_x = sum(set_of_values)
    mu = sum_of_x / n
    sigma = sum(x**2 for x in set_of_values) / n - mu
    return mu, sigma


def group_overlapping_with_same_class(map_of_regions, keep_singles=False):
    """merge overlapping rectangles with the same class
    Merge if there's overlapping between at leat 2 regions if keep_singles=False
    otherwise it keeps single rectangles.
    Args:
        map_of_regions:  {"label": [[rect1, p1], [rect2, p2], ..], "label2"...}
    """
    grouped_map = defaultdict(list)
    for label, rect_prob_list in map_of_regions.items():
        # extract rectangles from the array of pairs
        rects_only = np.array([value[0] for value in rect_prob_list])
        # group them
        factor = 2 if keep_singles else 1
        rect_list, _ = cv2.groupRectangles(
            rects_only.tolist() * factor, 1, eps=0.99)
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

            avg_prob = sum_of_prob / merged_count
            merged_rect_prob_list.append((merged_rect, avg_prob))

        if len(rect_list) > 0:
            grouped_map[label] = merged_rect_prob_list
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
        per_roi_probabilities = tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes]))
        # [tested positions, num_classes]

        # array[0]=values, [1]=indices
        top_k = tf.nn.top_k(per_roi_probabilities, k=TOP_K)
        # each with shape [tested_positions, k]
        original_image, eval_image = image_processing.get_original_and_processed_image(
            tf.constant(args.image_path),
            INPUT_SIDE,
            image_type=args.image_path.split('.')[-1])

        # roi placehoder
        roi_ = tf.placeholder(tf.uint8)
        # rop preprocessing, single image classification
        roi_preproc = image_processing.zm_mp(
            image_processing.resize_bl(
                tf.image.convert_image_dtype(roi_, tf.float32),
                pgnet.INPUT_SIDE))

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            input_images, image = sess.run([eval_image, original_image])
            image = cv2.normalize(
                image, image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            ####
            start = time.time()
            probability_map, top_values, top_indices = sess.run(
                [logits, top_k[0], top_k[1]],
                feed_dict={
                    "images_:0": input_images
                })
            nn_time = time.time() - start
            print("NN time: {}".format(nn_time))

            # scaling factor between original image and resized image
            full_image_scaling_factors = np.array(
                [image.shape[1] / INPUT_SIDE, image.shape[0] / INPUT_SIDE])

            # let's think to the net as a big net, with the last layer (before the FC
            # layers for classification) with a receptive field of
            # LAST_KERNEL_SIDE x LAST_KERNEL_SIDE. Lets approximate the net with this last kernel:
            # If the image is scaled down to  LAST_KERNEL_SIDE x LAST_KERNEL_SIDE
            # the output is a single point.
            # if the image is scaled down to something bigger
            # (that make the output side of contolution integer) the result is a spacial map
            # of points. Every point has a depth of num classes.

            # for every image in the input batch
            for _ in range(len(input_images)):
                probability_coords = 0
                glance = defaultdict(list)
                # select count(*), avg(prob) from map group by label, order by count, avg.
                group = defaultdict(lambda: defaultdict(float))
                for pmap_y in range(probability_map.shape[1]):
                    # calculate position in the downsampled image ds
                    ds_y = pmap_y * pgnet.CONV_STRIDE
                    for pmap_x in range(probability_map.shape[2]):
                        ds_x = pmap_x * pgnet.CONV_STRIDE

                        print(top_values[probability_coords][0])
                        if top_indices[probability_coords][
                                0] != BACKGROUND_CLASS and top_values[
                                    probability_coords][0] >= MIN_PROB:

                            # create coordinates of rect in the downsampled image
                            # convert to numpy array in order to use broadcast ops
                            coord = [ds_x, ds_y, ds_x + pgnet.LAST_KERNEL_SIDE,
                                     ds_y + pgnet.LAST_KERNEL_SIDE]
                            # if something is found, append rectagle to the
                            # map of rectalges per class
                            rect = upsample_and_shift(
                                coord, pgnet.DOWNSAMPLING_FACTOR, [0, 0],
                                full_image_scaling_factors)

                            prob = top_values[probability_coords][0]
                            label = PASCAL_LABELS[top_indices[
                                probability_coords][0]]

                            rect_prob = [rect, prob]
                            print('Glance: {} ({})'.format(rect_prob, label))
                            glance[label].append(rect_prob)
                            group[label]["count"] += 1
                            group[label]["prob"] += prob

                        # update probability coord value
                        probability_coords += 1

                classes = group.keys()
                print('Found {} classes: {}'.format(len(classes), classes))

                min_freq = LOOKED_POS
                min_prob = 1  # +/- 0.05
                eps = 0.01
                for label in group:
                    group[label]["prob"] /= group[label]["count"]
                    prob = group[label]["prob"]
                    freq = group[label]["count"]

                    if freq < min_freq:
                        min_freq = freq
                    if prob < min_prob:
                        min_prob = prob

                # pruning
                group = {
                    label: value
                    for label, value in group.items()
                    if value["prob"] > min_prob + eps and value["count"] >
                    min_freq
                }

                classes = group.keys()
                print('Remaining classes: {} ({})'.format(classes, len(
                    classes)))

                looked_pos = sum(value['count'] for value in group.values())
                print(
                    'New looked positions: {} vs avaiable positions {}'.format(
                        looked_pos, LOOKED_POS))

                print(group)
                print(
                    'Scale the probability of each class, using the relative frequency')
                rankmap = defaultdict(float)
                for label in group:
                    print(label, group[label]["prob"], group[label]["count"])
                    relative_freq = group[label]["count"] / looked_pos
                    print('RF ', relative_freq)
                    group[label]["prob"] *= relative_freq
                    print('NP: ', group[label]["prob"])
                    rankmap[label] = group[label]["prob"]

                # keep rectangles from local glance, only of the remaining labels
                glance = {label: value
                          for label, value in glance.items()
                          if label in classes}

                # merge overlapping rectangles for each class
                global_rect_prob = group_overlapping_with_same_class(glance)

                if False:
                    for label, rect_prob_list in global_rect_prob.items():
                        for rect_prob in rect_prob_list:
                            rect = rect_prob[0]
                            prob = rect_prob[1]
                            draw_box(
                                image,
                                rect,
                                "{} {:.3}".format(label, prob),
                                LABEL_COLORS[label],
                                thickness=1)

                # loop preserving order, because rois are evaluated in order
                rois = []
                for label, value in sorted(
                        rankmap.items(), key=operator.itemgetter(1),
                        reverse=True):
                    # extract rectangles for each image and classify it.
                    # if the classification gives the same global label as top-1(2,3?) draw it
                    # else skip it.
                    for rect_prob in global_rect_prob[label]:
                        rect = rect_prob[0]
                        y2 = rect[3]
                        y1 = rect[1]
                        x2 = rect[2]
                        x1 = rect[0]
                        roi = image[y1:y2, x1:x2]

                        rois.append(
                            sess.run(roi_preproc, feed_dict={roi_: roi}))

                # evaluate top values for every image in the batch of rois
                top_values, top_indices = sess.run(
                    [top_k[0], top_k[1]], feed_dict={"images_:0": rois})

                roi_id = 0
                draw_rect_prob = defaultdict(list)
                for label, confidence in sorted(
                        rankmap.items(), key=operator.itemgetter(1),
                        reverse=True):

                    # evaluate only regions with confidence greater then threshold
                    # rois are collected in order of highest confidence of the global label
                    if confidence > eps:
                        # loop over rect with the current label
                        for rect_prob in global_rect_prob[label]:
                            roi_prob = top_values[roi_id][0]
                            roi_label = PASCAL_LABELS[top_indices[roi_id][0]]

                            print(
                                'roi_prob {} vs {}, roi_label {} vs {}'.format(
                                    roi_prob, rect_prob[1], roi_label, label))
                            print(label, confidence)
                            print(top_values[roi_id],
                                  [PASCAL_LABELS[top_indices[roi_id][l]]
                                   for l in range(TOP_K)])

                            draw = False
                            draw_label = ''
                            draw_prob = 0

                            if abs(roi_prob - rect_prob[1]) <= max(
                                    rankmap[roi_label], confidence) + eps:
                                top_labels = [
                                    PASCAL_LABELS[top_indices[roi_id][lbl]]
                                    for lbl in range(1, TOP_K)
                                ]

                                if label == roi_label:
                                    draw = True
                                    draw_prob = max(rect_prob[1], roi_prob)
                                    draw_label = label
                                elif label in top_labels:
                                    draw = True
                                    if rankmap[label] > rankmap[roi_label]:
                                        draw_prob = rect_prob[1]
                                        draw_label = label
                                    else:
                                        draw_prob = roi_prob
                                        draw_label = roi_label
                            if draw:
                                print('winner: {} {}'.format(draw_label,
                                                             draw_prob))
                                draw_rect_prob[draw_label].append(
                                    [rect_prob[0], draw_prob])
                            roi_id += 1
                    else:
                        # exploit the sorted confidence to speed up evaluaion
                        break

                # keep singles
                draw_rect_prob = group_overlapping_with_same_class(
                    draw_rect_prob, keep_singles=True)
                for label, rect_prob_list in draw_rect_prob.items():
                    for rect_prob in rect_prob_list:
                        draw_box(
                            image,
                            rect_prob[0],
                            "{}({:.3})".format(label, rect_prob[1]),
                            LABEL_COLORS[label],
                            thickness=2)

                cv2.imshow("img", image)
                #legend()
                cv2.waitKey(0)
                sys.exit()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Apply the model to image-path")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--image-path")
    sys.exit(main(PARSER.parse_args()))
