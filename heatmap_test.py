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
PATCH_SIDE = pgnet.INPUT_SIDE + pgnet.DOWNSAMPLING_FACTOR * 3
NO_PATCHES_PER_SIDE = 4
#eg: 768 -> 4 patch 192*192 -> each one produces a spatial map of 4x4x20 probabilities
RESIZED_INPUT_SIDE = PATCH_SIDE * NO_PATCHES_PER_SIDE

# trained pgnet constants
BACKGROUND_CLASS = 20
MIN_PROB = 0.4


def batchify_image(image_path, image_type="jpg"):
    """Return the original image as read from image_path and the image splitted as a batch tensor.
    Args:
        image_path: image path
        image_type: image type
    Returns:
        original_image, patches
        where original image is a tensor in the format [widht, height 3]
        and patches is a tensor of processed images, ready to be classified, with size
        [batch_size, w, h, 3]"""

    if image_type == "jpg":
        original_image = image_processing.read_image_jpg(image_path, 3)
    else:
        original_image = image_processing.read_image_png(image_path, 3)

    resized_image = image_processing.resize_bl(original_image,
                                               RESIZED_INPUT_SIDE)
    """
    # OK OK TODO: test if whitening every patch improves results
    resized_image = image_processing.zm_mp(resized_image)
    print(resized_image)

    # extract 4 patches
    resized_image = tf.expand_dims(resized_image, 0)
    print(resized_image)
    patches = tf.space_to_depth(resized_image, PATCH_SIDE)
    print(patches)  #1,4,4,192*192*3
    patches = tf.reshape(patches,
                         [NO_PATCHES_PER_SIDE**2, PATCH_SIDE, PATCH_SIDE, 3])
    print(patches)
    #resized_image = image_processing.zm_mp(resized_image)
    return tf.image.convert_image_dtype(original_image, tf.uint8), patches
    
    """
    resized_image = tf.expand_dims(resized_image, 0)
    patches = tf.space_to_depth(resized_image, PATCH_SIDE)
    print(patches)
    patches = tf.squeeze(patches, [0])  #4,4,192*192*3
    print(patches)
    patches = tf.reshape(patches,
                         [NO_PATCHES_PER_SIDE**2, PATCH_SIDE, PATCH_SIDE, 3])
    print(patches)
    patches_a = tf.split(0, NO_PATCHES_PER_SIDE**2, patches)
    print(patches_a)
    normalized_patches = []
    for patch in patches_a:
        patch_as_input_image = image_processing.zm_mp(tf.reshape(
            tf.squeeze(patch, [0]), [PATCH_SIDE, PATCH_SIDE, 3]))
        print(patch_as_input_image)
        normalized_patches.append(patch_as_input_image)
    batch_of_patches = tf.pack(normalized_patches)
    return tf.image.convert_image_dtype(original_image,
                                        tf.uint8), batch_of_patches


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
        print(logits)
        # each cell in coords (batch_position, i, j) -> is a probability vector
        per_batch_probabilities = tf.nn.softmax(tf.reshape(logits,
                                                           [-1, num_classes]))
        # [tested positions, num_classes]
        print(per_batch_probabilities)
        #sys.exit(1)

        # array[0]=values, [1]=indices
        top_k = tf.nn.top_k(per_batch_probabilities, k=5)
        # each with shape [tested_positions, k]

        original_image, batch = batchify_image(
            tf.constant(args.image_path),
            image_type=args.image_path.split('.')[-1])

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            batchifyed_image = batch.eval()

            #for idx, img in enumerate(batchifyed_image):
            #    cv2.imshow(str(idx), img)
            #cv2.waitKey(0)

            probability_map, top_values, top_indices, image = sess.run(
                [logits, top_k[0], top_k[1], original_image],
                feed_dict={
                    "images_:0": batchifyed_image
                })
            print("Predictions: ", probability_map.size, probability_map.shape)

            # extract image (resized image) dimensions to get the scaling factor
            # respect to the original image
            print(image.shape)
            original_scaling_factor_x = image.shape[0] / RESIZED_INPUT_SIDE
            original_scaling_factor_y = image.shape[1] / RESIZED_INPUT_SIDE

            print(original_scaling_factor_x, original_scaling_factor_y)
            print(top_values)
            print(top_values.shape)

            # let's think to the net as a big net, with the last layer (before the FC
            # layers for classification) with a receptive field of
            # LAST_KERNEL_SIDE x LAST_KERNEL_SIDE. Lets approximate the net with this last kernel:
            # If the image is scaled down to  LAST_KERNEL_SIDE x LAST_KERNEL_SIDE
            # the output is a single point.
            # if the image is scaled down to something bigger
            # (that make the output side of contolution integer) the result is a spacial map
            # of points. Every point has a depth of num classes.

            # save coordinates and batch id, format: [batch_id, y1, x1, y2, x2]
            batch_id = 0
            coords = []
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

                            # convert to numpy array in order to use broadcast ops
                            # create coordinates of rect in the downsampled image
                            coord = np.array([batch_id, ds_y, ds_x,
                                              ds_y + pgnet.LAST_KERNEL_SIDE,
                                              ds_x + pgnet.LAST_KERNEL_SIDE])
                            coords.append(coord)

                            # if something is found, append rectagle to the 
                            # map of rectalges per class
                            print(coord)

                            if top_values[probability_coords][
                                    0] > MIN_PROB and top_indices[
                                        probability_coords][
                                            0] != BACKGROUND_CLASS:

                                top_1_label = PASCAL_LABELS[top_indices[
                                    probability_coords][0]]

                                print(coord[1:])

                                # upsample coordinates to find the coordinates of the cell
                                box = coord[1:] * pgnet.DOWNSAMPLING_FACTOR
                                print(box)

                                # shift coordinates to the position of the current cell
                                # in the resized input image
                                box += [PATCH_SIDE * i, PATCH_SIDE * j,
                                        PATCH_SIDE * i, PATCH_SIDE * j]
                                print(box)

                                # scale coordinates to the input image
                                input_box = np.ceil(
                                    box *
                                    [original_scaling_factor_x,
                                     original_scaling_factor_y,
                                     original_scaling_factor_x,
                                     original_scaling_factor_y]).astype(int)
                                print(input_box)
                                # convert tf rect format to opencv rect format
                                #xmin, ymin, xmax, ymax
                                cv_rect = [input_box[1], input_box[0],
                                           input_box[3], input_box[2]]
                                # save the probability associated to the rect
                                # [ [rect], probability]
                                input_image_coords[top_1_label].append(
                                    [cv_rect,
                                     top_values[probability_coords][0]])
                            # update probability coord value
                            probability_coords += 1

                    batch_id += 1

            print(batch_id, probability_coords)
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
                    cv2.putText(image,
                                label, (rect[0] + 10, rect[1] + 10),
                                0,
                                1,
                                color,
                                thickness=2)
            cv2.imshow("img", image)
            cv2.waitKey(0)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Apply the model to image-path")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--image-path")
    sys.exit(main(PARSER.parse_args()))
