#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Utils contains utility functions to test the model"""

import math
from collections import defaultdict
import cv2
import numpy as np
from inputs import pascal


def rnd_color():
    """ Generate random colors in RGB format"""
    rnd = lambda: np.random.randint(0, 255)
    return (rnd(), rnd(), rnd())


LABEL_COLORS = {label: rnd_color() for label in pascal.CLASSES}


def legend():
    """Display a box containing the associations between
    colors and labels"""
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    height = 20
    for label in pascal.CLASSES:
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
    input_box = np.ceil(box * [
        scaling_factor_x, scaling_factor_y, scaling_factor_x, scaling_factor_y
    ]).astype(int)
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


def group_overlapping_regions(map_of_regions, eps=0.9):
    """ Clusters regions with same classe basing it on the rectangle similarity
    Args:
        map_of_regions:  {"label": [[rect1, p1], [rect2, p2], ..], "label2"...}
        eps: used only if CLUSTERING is used. Is the region similarity threshold
    """

    grouped_map = defaultdict(list)
    for label, rect_prob_list in map_of_regions.items():
        # extract rectangles from the array of pairs
        rects_only = np.array([value[0] for value in rect_prob_list])

        # group them
        rect_list, _ = cv2.groupRectangles(rects_only.tolist(), 1, eps=eps)

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
