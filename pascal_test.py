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

    # export model.pb from session dir. Skip if model.pb already exists
    train.export_model()

    current_dir = os.path.abspath(os.getcwd())
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

        # exteact the pgnet output from the graph and scale the result
        # using softmax
        softmax_linear = graph.get_tensor_by_name("softmax_linear/out:0")
        # softmax_linear is the output of a 1x1xNUM_CLASS convolution
        # to use the softmax we have to reshape it back to (?,NUM_CLASS)
        softmax_linear = tf.reshape(softmax_linear, [-1, pgnet.NUM_CLASS])
        softmax = tf.nn.softmax(softmax_linear, name="softmax")

        # get the input queue of resized (or cropped) test images
        # use 29 as batch_size because is a divisor of the test dataset size
        #test_center_cropped_queue, test_center_cropped_filename_queue = pascal_input.test(
        #    args.test_ds,
        #    29,
        #    args.test_ds + "/ImageSets/Main/test.txt",
        #    method="central-crop")

        test_resize_queue, test_resize_filename_queue = pascal_input.test(
            args.test_ds,
            29,
            args.test_ds + "/ImageSets/Main/test.txt",
            method="resize")

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
                    #image_batch_cropped, filename_batch_cropped = sess.run(
                    #    [test_center_cropped_queue,
                    #     test_center_cropped_filename_queue])

                    # run predction on images central cropped (or padded)
                    #batch_predictions_cropped = sess.run(
                    #    softmax,
                    #    feed_dict={
                    #        "keep_prob_:0": 1.0,
                    #        "images_:0": image_batch_cropped,
                    #    })

                    image_batch_resize, filename_batch_resize = sess.run(
                        [test_resize_queue, test_resize_filename_queue])

                    # run prediction on images resized
                    batch_predictions_resize = sess.run(softmax,
                                                    feed_dict={
                                                        "keep_prob_:0": 1.0,
                                                        "images_:0":
                                                        image_batch_resize,
                                                    })

                    for batch_elem_id, prediction_probs in enumerate(
                            #batch_predictions_cropped):
                            batch_predictions_resize):
                        decoded_filename = filename_batch_resize[
                            batch_elem_id].decode("utf-8")
                        print(decoded_filename)
                        for idx, pred in enumerate(prediction_probs):
                            #avg_pred = (
                            #    pred + batch_predictions_resize[batch_elem_id][idx]
                            #) / 2
                            avg_pred = pred
                            files[build_trainval.CLASSES[idx]].write(
                                "{} {}\n".format(decoded_filename, avg_pred))

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
    parser = argparse.ArgumentParser(description="Test the model")
    parser.add_argument("--device", default="/gpu:1")
    parser.add_argument("--test-ds")
    sys.exit(main(parser.parse_args()))
