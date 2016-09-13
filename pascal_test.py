#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""./pascal_test.py PASCAL_2012_test_dataset/VOCdevkit/VOC2012

Reads the file list in: argv[1]/ImageSets/Main/test.txt


Image classification competition resuts:
    for every of the 20 classes, creates a file with the following format:

    ```
    filename: comp1_cls_test_<class>.txt:
    <filename (no extension)> confidence // eg: 2009_000001 0.0563131
    ```
    The results are placed in the directory `results/VOC2012/Main/`
"""

import argparse
import os
import sys
import tensorflow as tf
import pgnet
import train
import pascal_input
import build_trainval


def main(args):
    """ main """

    if not os.path.exists(args.test_ds):
        print("{} does not exists".format(args.test_ds))
        return 1

    current_dir = os.path.abspath(os.getcwd())

    # Number of classes in the dataset plus 1.
    # Labelp pascal_input. NUM_CLASSES + 1 is reserved for
    # an (unused) background class.
    num_classes = pascal_input.NUM_CLASSES + 1

    # export model.pb from session dir. Skip if model.pb already exists
    pgnet.export_model(num_classes, current_dir + "/session", "model-0",
                       "model.pb")

    results_dir = "{}/results".format(current_dir)

    ##### Image classification competition #####
    # open a file for every class
    files = {label: open(
        results_dir + "/VOC2012/Main/comp1_cls_test_{}.txt".format(label), "w")
             for label in build_trainval.CLASSES}

    with tf.Graph().as_default() as graph, tf.device(args.device):
        const_graph_def = tf.GraphDef()
        with open(train.TRAINED_MODEL_FILENAME, 'rb') as saved_graph:
            const_graph_def.ParseFromString(saved_graph.read())
            # replace current graph with the saved graph def (and content)
            # name="" is importat because otherwise (with name=None)
            # the graph definitions will be prefixed with import.
            tf.import_graph_def(const_graph_def, name="")

        # now the current graph contains the trained model

        # input placeholder
        images_ = graph.get_tensor_by_name(pgnet.INPUT_TENSOR_NAME + ":0")
        logits = graph.get_tensor_by_name(pgnet.OUTPUT_TENSOR_NAME + ":0")
        # (?, n, n, NUM_CLASSES) tensor
        # each cell in coords (batch_position, i, j) -> is a probability vector
        per_region_probabilities = tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes]))
        # logits is the output of a 1x1xNUM_CLASS convolution

        # classification by localization
        k = 1
        input_side = pgnet.INPUT_SIDE + pgnet.LAST_CONV_INPUT_STRIDE * pgnet.DOWNSAMPLING_FACTOR * k

        # get the input queue of resized test images
        # use 29 as batch_size because is a proper divisor of the test dataset size
        test_queue, test_filename_queue = pascal_input.test(
            args.test_ds, 29, input_side,
            args.test_ds + "/ImageSets/Main/test.txt")

        init_op = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:

            sess.run(init_op)

            # Start input enqueue threads.
            print("Starting input enqueue threads. Please wait...")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                processed = 0
                while not coord.should_stop():
                    # extract batches from queues
                    image_batch, filename_batch = sess.run(
                        [test_queue, test_filename_queue])

                    # run prediction on images resized
                    batch_predictions = sess.run(
                        softmax, feed_dict={images_: image_batch})

                    for batch_elem_id, prediction_probs in enumerate(
                            #batch_predictions_cropped):
                            batch_predictions):
                        decoded_filename = filename_batch[
                            batch_elem_id].decode("utf-8")
                        print(decoded_filename)
                        for idx, pred in enumerate(prediction_probs):
                            files[build_trainval.CLASSES[idx]].write(
                                "{} {}\n".format(decoded_filename, pred))

                        processed += 1

            except tf.errors.OutOfRangeError:
                print("[I] Done. Test completed!")
                print("Processed {} images".format(processed))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

        # close all files
        for label in files:
            files[label].close()


if __name__ == "__main__":
    # pylint: disable=C0103
    PARSER = argparse.ArgumentParser(description="Test the model")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--test-ds")
    sys.exit(main(PARSER.parse_args()))
