#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""./localization_via_frequencies.py --image-path <img path>"""

import argparse
import os
import sys
import math
from collections import defaultdict
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import train
import pgnet
import pascal_input

# pascal sorted labels
PASCAL_LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                 "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
                 "motorbike", "person", "pottedplant", "sheep", "sofa",
                 "train", "tvmonitor"]

# detection constants
RECT_SIMILARITY = 0.9

# trained pgnet constants
BACKGROUND_CLASS = 20


def rnd_color():
    """ Generate random colors in RGB format"""
    rnd = lambda: np.random.randint(0, 255)
    return (rnd(), rnd(), rnd())


LABEL_COLORS = {label: rnd_color() for label in PASCAL_LABELS}


def legend():
    """Display a box containing the associations between
    colors and labels"""
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    height = 20
    for label in PASCAL_LABELS:
        color = LABEL_COLORS[label]
        cv2.putText(image, label, (5, height), 0, 1, color, thickness=2)
        height += 20

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
        label, (rect[0] + 15, rect[1] + 15),
        0,
        1,
        color,
        thickness=thickness)


def intersection(rect_a, rect_b):
    """Returns the intersection of rect_a and rect_b."""

    left = min(rect_a[0], rect_b[0])
    top = min(rect_a[1], rect_b[1])

    a_width = abs(rect_a[0] - rect_a[2])
    b_width = abs(rect_b[0] - rect_b[2])
    a_height = abs(rect_a[1] - rect_a[3])
    b_height = abs(rect_b[1] - rect_b[3])

    right = min(rect_a[0] + a_width, rect_b[0] + b_width)
    bottom = min(rect_a[1] + a_height, rect_b[1] + b_height)

    width = right - left
    height = bottom - top

    if left <= right and top <= bottom:
        return (left, top, width, height)
    return ()


def intersect(rect_a, rect_b):
    """Returns true if rect_a intersects rect_b"""
    return intersection(rect_a, rect_b) != ()


def merge(rect_a, rect_b):
    """Returns the merge of rect_a and rect_b"""
    left = min(rect_a[0], rect_b[0])
    top = min(rect_a[1], rect_b[1])
    right = max(rect_a[0] + abs(rect_a[0] - rect_a[2]),
                rect_b[0] + abs(rect_b[0] - rect_b[2]))
    bottom = max(rect_a[1] + abs(rect_a[1] - rect_a[3]),
                 rect_b[1] + abs(rect_b[1] - rect_b[3]))
    return (left, top, right - left, bottom - top)


def norm(point):
    """Returns sqrt(point[0]**2 + point[1]**2)"""
    return math.sqrt(point[0]**2 + point[1]**2)


def l2_distance(point_a, point_b):
    """Returns norm((point_a[0] - point_b[0], point_a[1] - point_b[1]))"""
    return norm((point_a[0] - point_b[0], point_a[1] - point_b[1]))


def merge_overlapping(rects_probs):
    tot = len(rects_probs)
    skip_idx = []
    merged_rects_probs = defaultdict(list)

    for i in range(0, tot - 1):
        merged = 0
        for j in range(i + 1, tot):
            if j not in skip_idx and intersect(rects_probs[i][0],
                                               rects_probs[j][0]):
                skip_idx.append(j)
                if len(merged_rects_probs[i]) == 0:
                    merged_rects_probs[i] = rects_probs[i]
                merged_rects_probs[i][0] = merge(merged_rects_probs[i][0],
                                                 rects_probs[j][0])
                merged_rects_probs[i][1] += rects_probs[j][1]
                merged += 1

        # consider first rectangle (centroid)
        merged += 1
        if merged > 1:
            merged_rects_probs[i][1] /= merged
            merged_rects_probs[i].append(merged)

    return merged_rects_probs.values()


def group_overlapping_with_same_class(map_of_regions):
    """merge overlapping rectangles with the same class.
    Merge only rectangles with at lest threshold intersections.
    Args:
        map_of_regions:  {"label": [[rect1, p1], [rect2, p2], ..], "label2"...}
    """
    grouped_map = defaultdict(list)
    for label, rect_prob_list in map_of_regions.items():
        # extract rectangles from the array of pairs
        rects_only = np.array([value[0] for value in rect_prob_list])

        # group them
        rect_list, _ = cv2.groupRectangles(
            rects_only.tolist(), 1, eps=RECT_SIMILARITY)

        if len(rect_list) > 0:
            # for every merged rectangle, calculate the avg prob and
            # the number of intersection
            merged_rect_infos = []
            for merged_rect in rect_list:
                sum_of_prob = 0.0
                merged_count = 0
                for idx, original_rect in enumerate(rects_only):
                    if intersect(original_rect, merged_rect):
                        sum_of_prob += rect_prob_list[idx][1]
                        merged_count += 1

                if merged_count > 0:
                    avg_prob = sum_of_prob / merged_count
                    merged_rect_infos.append(
                        (merged_rect, avg_prob, merged_count))

            grouped_map[label] = merged_rect_infos

    return grouped_map


def main(args):
    """ main """

    if not os.path.exists(args.test_ds):
        print("{} does not exists".format(args.test_ds))
        return 1

    current_dir = os.path.abspath(os.getcwd())

    # Number of classes in the dataset plus 1.
    # Labelp pascal_input. NUM_CLASSES + 1 is reserved for
    # the background class.
    num_classes = pascal_input.NUM_CLASSES + 1

    # export model.pb from session dir. Skip if model.pb already exists
    pgnet.export_model(num_classes, current_dir + "/session", "model-0",
                       "model.pb")

    results_dir = "{}/results".format(current_dir)
    files = {label: open(
        results_dir + "/VOC2012/Main/comp1_cls_test_{}.txt".format(label), "w")
             for label in PASCAL_LABELS}

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
        images_ = graph.get_tensor_by_name(pgnet.INPUT_TENSOR_NAME + ":0")
        # each cell in coords (batch_position, i, j) -> is a probability vector
        per_region_probabilities = tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes]))
        # [tested positions, num_classes]

        # array[0]=values, [1]=indices
        # get every probabiliy, because we can use localization to do classification
        top_k = tf.nn.top_k(per_region_probabilities, k=num_classes)
        # each with shape [tested_positions, k]

        k = 2
        input_side = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * pgnet.LAST_CONV_INPUT_STRIDE * k

        test_queue, test_filename_queue = pascal_input.test(
            args.test_ds, 29, input_side,
            args.test_ds + "/ImageSets/Main/test.txt")

        init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables())

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            sess.run(init_op)
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(
                sess=sess, coord=coordinator)

            try:
                processed = 0
                while not coordinator.should_stop():
                    image_batch, filename_batch = sess.run(
                        [test_queue, test_filename_queue])

                    probability_map, top_values, top_indices = sess.run(
                        [logits, top_k[0], top_k[1]],
                        feed_dict={images_: image_batch})

                    # let's think to the net as a big net, with the last layer (before the FC
                    # layers for classification) with a receptive field of
                    # LAST_KERNEL_SIDE x LAST_KERNEL_SIDE. Lets approximate the net with this last kernel:
                    # If the image is scaled down to  LAST_KERNEL_SIDE x LAST_KERNEL_SIDE
                    # the output is a single point.
                    # if the image is scaled down to something bigger
                    # (that make the output side of contolution integer) the result is a spacial map
                    # of points. Every point has a depth of num classes.

                    # for every image in the input batch
                    for batch_elem_id in range(len(image_batch)):
                        # scaling factor between original image and resized image
                        decoded_filename = filename_batch[
                            batch_elem_id].decode("utf-8")
                        image = Image.open(args.test_ds + "/JPEGImages/" +
                                           decoded_filename + ".jpg")
                        full_image_scaling_factors = np.array(
                            [image.size[0] / input_side,
                             image.size[1] / input_side])

                        probability_coords = 0
                        glance = defaultdict(list)
                        # select count(*), avg(prob) from map group by label, order by count, avg.
                        group = defaultdict(lambda: defaultdict(float))
                        for pmap_y in range(probability_map.shape[1]):
                            # calculate position in the downsampled image ds
                            ds_y = pmap_y * pgnet.LAST_CONV_OUTPUT_STRIDE
                            for pmap_x in range(probability_map.shape[2]):
                                ds_x = pmap_x * pgnet.LAST_CONV_OUTPUT_STRIDE

                                if top_indices[probability_coords][
                                        0] != BACKGROUND_CLASS:

                                    # create coordinates of rect in the downsampled image
                                    # convert to numpy array in order to use broadcast ops
                                    coord = [ds_x, ds_y,
                                             ds_x + pgnet.LAST_KERNEL_SIDE,
                                             ds_y + pgnet.LAST_KERNEL_SIDE]
                                    # if something is found, append rectagle to the
                                    # map of rectalges per class
                                    rect = upsample_and_shift(
                                        coord, pgnet.DOWNSAMPLING_FACTOR,
                                        [0, 0], full_image_scaling_factors)

                                    prob = top_values[probability_coords][0]
                                    label = PASCAL_LABELS[top_indices[
                                        probability_coords][0]]

                                    rect_prob = [rect, prob]
                                    glance[label].append(rect_prob)
                                    group[label]["count"] += 1
                                    group[label]["prob"] += prob

                                # update probability coord value
                                probability_coords += 1

                        classes = group.keys()
                        print('Found {} classes: {}'.format(
                            len(classes), classes))

                        # find out the minimum amount of intersection among regions
                        # in the original image, that can be used to trigger a match
                        looked_pos_side = probability_map.shape[
                            1]  # or 2, is s square. 0 dim is batch
                        looked_pos = looked_pos_side**2

                        # Save the relative frequency for every class
                        # To trigger a match, at least a fraction of intersection should be present
                        for label in group:
                            group[label]["rf"] = group[label][
                                "count"] / looked_pos
                            files[label].write("{} {}\n".format(
                                decoded_filename, group[label]["rf"]))
                        for label in set(set(PASCAL_LABELS) - group.keys()):
                            files[label].write("{} {}\n".format(
                                decoded_filename, 0))

                        processed += 1

            except tf.errors.OutOfRangeError:
                print("[I] Done. Test completed!")
                print("Processed {} images".format(processed))
            finally:
                coordinator.request_stop()

            coordinator.join(threads)

        for label in files:
            files[label].close()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Test the model")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--test-ds")
    sys.exit(main(PARSER.parse_args()))
