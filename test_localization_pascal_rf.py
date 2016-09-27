#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""./test_localization_pascal_rf.py --test-ds ~/data/PASCAL_2012/test/VOCdevkit/VOC2012/"""

import argparse
import os
import sys
from collections import defaultdict
import tensorflow as tf
import numpy as np
import train
import utils
from pgnet import model
from inputs import pascal, image_processing

# detection parameters
RECT_SIMILARITY = 0.9


def main(args):
    """ main """

    if not os.path.exists(args.test_ds):
        print("{} does not exists".format(args.test_ds))
        return 1

    # export model.pb from session dir. Skip if model.pb already exists
    model.export(train.NUM_CLASSES, train.SESSION_DIR, "model-best-0",
                 train.MODEL_PATH)

    results_dir = "{}/results".format(
        os.path.dirname(os.path.abspath(__file__)))
    files = {label: open(
        results_dir + "/VOC2012/Main/comp3_det_test_{}.txt".format(label), "w")
             for label in pascal.CLASSES}

    graph = model.load(train.MODEL_PATH, args.device)
    with graph.as_default():
        # (?, n, n, NUM_CLASSES) tensor
        logits = graph.get_tensor_by_name(model.OUTPUT_TENSOR_NAME + ":0")
        images_ = graph.get_tensor_by_name(model.INPUT_TENSOR_NAME + ":0")
        # each cell in coords (batch_position, i, j) -> is a probability vector
        per_region_probabilities = tf.nn.softmax(
            tf.reshape(logits, [-1, train.NUM_CLASSES]))
        # [tested positions, train.NUM_CLASSES]

        # array[0]=values, [1]=indices
        # get every probabiliy, because we can use localization to do classification
        top_k = tf.nn.top_k(per_region_probabilities, k=train.NUM_CLASSES)
        # each with shape [tested_positions, k]

        k = 2
        input_side = model.INPUT_SIDE + model.DOWNSAMPLING_FACTOR * model.LAST_CONV_INPUT_STRIDE * k

        test_queue, test_filename_queue = pascal.test(
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
                    probability_coords = 0
                    for batch_elem_id in range(len(image_batch)):
                        # scaling factor between original image and resized image
                        decoded_filename = filename_batch[
                            batch_elem_id].decode("utf-8")

                        image = sess.run(
                            image_processing.read_image_jpg(
                                args.test_ds + "/JPEGImages/" +
                                decoded_filename + ".jpg"))
                        full_image_scaling_factors = np.array(
                            [image.shape[1] / input_side,
                             image.shape[0] / input_side])

                        glance = defaultdict(list)

                        group = defaultdict(lambda: defaultdict(float))
                        for pmap_y in range(probability_map.shape[1]):
                            # calculate position in the downsampled image ds
                            ds_y = pmap_y * model.LAST_CONV_OUTPUT_STRIDE
                            for pmap_x in range(probability_map.shape[2]):
                                ds_x = pmap_x * model.LAST_CONV_OUTPUT_STRIDE

                                if top_indices[probability_coords][
                                        0] != pascal.BACKGROUND_CLASS_ID:

                                    # create coordinates of rect in the downsampled image
                                    # convert to numpy array in order to use broadcast ops
                                    coord = [ds_x, ds_y,
                                             ds_x + model.LAST_KERNEL_SIDE,
                                             ds_y + model.LAST_KERNEL_SIDE]
                                    # if something is found, append rectagle to the
                                    # map of rectalges per class
                                    rect = utils.upsample_and_shift(
                                        coord, model.DOWNSAMPLING_FACTOR,
                                        [0, 0], full_image_scaling_factors)

                                    prob = top_values[probability_coords][0]
                                    label = pascal.CLASSES[top_indices[
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
                        # or 2, is s square. 0 dim is batch
                        map_side = probability_map.shape[1]
                        map_area = map_side**2

                        min_intersection = map_side

                        # Save the relative frequency for every class
                        # To trigger a match, at least a fraction of intersection should be present
                        for label in group:
                            group[label]["prob"] /= group[label]["count"]
                            group[label]["rf"] = group[label][
                                "count"] / map_area

                        # merge overlapping rectangles for each class.
                        # return a map of {"label": [rect, prob, count]
                        localize = utils.group_overlapping_regions(
                            glance, eps=RECT_SIMILARITY)

                        detected_labels = set()
                        for label, rect_prob_list in localize.items():
                            for rect_prob in rect_prob_list:
                                count = rect_prob[2]
                                freq = group[label]["rf"]
                                if count >= min_intersection and freq > 0.1:
                                    detected_labels.add(label)
                                    confidence = rect_prob[1]
                                    rect = rect_prob[0]
                                    left = rect[0]
                                    top = rect[1]
                                    right = rect[2]
                                    bottom = rect[3]
                                    files[label].write(
                                        "{} {} {} {} {} {}\n".format(
                                            decoded_filename, confidence, left,
                                            top, right, bottom))

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
