#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Usage: ./pascifar_test.py --device <device> --test-ds pascifar_path
Runs the trained model on the PASCIFAR dataset that is a classification dataset.
Prints the accuracy for each class and the average accuracy.
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import train
import pgnet
import pascifar_input
import pascal_input

BATCH_SIZE = 100


def main(args):
    """ main """

    if not os.path.exists(args.test_ds):
        print("{} does not exists".format(args.test_ds))
        return 1

    # export model.pb from session dir. Skip if model.pb already exists
    current_dir = os.path.abspath(os.getcwd())

    # Number of classes in the dataset plus 1.
    # Labelp pascal_input. NUM_CLASSES + 1 is reserved for
    # an (unused) background class.
    num_classes = pascal_input.NUM_CLASSES + 1

    pgnet.export_model(num_classes, current_dir + "/session", "model-0",
                       "model.pb")

    graph = tf.Graph()
    with graph.as_default(), tf.device(args.device):
        const_graph_def = tf.GraphDef()
        with open(train.TRAINED_MODEL_FILENAME, 'rb') as saved_graph:
            const_graph_def.ParseFromString(saved_graph.read())
            # replace current graph with the saved graph def (and content)
            # name="" is importat because otherwise (with name=None)
            # the graph definitions will be prefixed with import.
            tf.import_graph_def(const_graph_def, name="")

        # now the current graph contains the trained model

        logits = graph.get_tensor_by_name(pgnet.OUTPUT_TENSOR_NAME + ":0")
        logits = tf.squeeze(logits, [1, 2])

        # sparse labels, pgnet output -> 20 possible values
        labels_ = tf.placeholder(tf.int64, [None])

        predicted_labels = tf.argmax(logits, 1)

        top_1_op = tf.nn.in_top_k(logits, labels_, 1)
        top_5_op = tf.nn.in_top_k(logits, labels_, 5)

        image_queue, label_queue = pascifar_input.test(
            args.test_ds, BATCH_SIZE, args.test_ds + "/ts.csv")

        # initialize all variables
        """
        all_variables = set(tf.all_variables())

        # EXCEPT model variables (that has been marked as trainable)
        model_variables = set(tf.trainable_variables())
        init_op = tf.initialize_variables(list(all_variables -
                                               model_variables))
        """
        init_op = tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(init_op)

            # Start input enqueue threads.
            print("Starting input enqueue threads. Please wait...")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                count_top_1 = 0.0
                count_top_5 = 0.0
                processed = 0
                #sum_accuracy_per_class = {label: 0.0
                #                          for label in PASCAL2PASCIFAR.values()
                #                          }
                while not coord.should_stop():
                    image_batch, label_batch = sess.run(
                        [image_queue, label_queue])

                    top_1, top_5, pl = sess.run(
                        [top_1_op, top_5_op, predicted_labels],
                        feed_dict={
                            "images_:0": image_batch,
                            labels_: label_batch,
                        })
                    count_top_1 += np.sum(top_1)
                    count_top_5 += np.sum(top_5)
                    processed += 1
                    print(pl)

                    print(label_batch)
                    print(top_1, top_5)

            except tf.errors.OutOfRangeError:
                total_sample_count = processed * BATCH_SIZE
                precision_at_1 = count_top_1 / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count

                print('precision @ 1 = {} recall @ 5 = {} [{} examples]'.
                      format(precision_at_1, recall_at_5, total_sample_count))
                print("Accuracy per class: ")
                # TODO
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Test the model using the PASCIFAR datast")
    PARSER.add_argument("--device", default="/gpu:1")
    PARSER.add_argument("--test-ds")
    sys.exit(main(PARSER.parse_args()))
