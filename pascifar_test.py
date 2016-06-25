#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Usage: ./pascifar_test.py PASCIFAR
Runs the trained model on the PASCIFAR dataset that is a classification dataset.
Prints the accuracy for each class and the average accuracy.
"""

import argparse
import os
import sys
import tensorflow as tf
import train
import pgnet
import pascifar_input
import pascal_input

BATCH_SIZE = 50


def main(args):
    """ main """

    if not os.path.exists(args.test_ds):
        print("{} does not exists".format(args.test_ds))
        return 1

    # export model.pb from session dir. Skip if model.pb already exists
    current_dir = os.path.abspath(os.getcwd())
    pgnet.export_model(pascal_input.NUM_CLASSES, current_dir + "/session",
                       "model-0", "model.pb")

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        const_graph_def = tf.GraphDef()
        with open(train.TRAINED_MODEL_FILENAME, 'rb') as saved_graph:
            const_graph_def.ParseFromString(saved_graph.read())
            # replace current graph with the saved graph def (and content)
            # name="" is importat because otherwise (with name=None)
            # the graph definitions will be prefixed with import.
            tf.import_graph_def(const_graph_def, name="")

        # now the current graph contains the trained model

        logits = graph.get_tensor_by_name(pgnet.OUTPUT_TENSOR_NAME + ":0")

        reshaped_logits = tf.squeeze(logits, [1, 2])

        # [batch_size] vector
        predictions = tf.argmax(reshaped_logits, 1)
        # sparse labels, pgnet output -> 20 possible values
        labels_ = tf.placeholder(tf.int64, [None])
        correct_predictions = tf.equal(labels_, predictions)

        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        image_queue, label_queue = pascifar_input.test(
            args.test_ds, BATCH_SIZE, args.test_ds + "/ts.csv")

        # initialize all variables
        all_variables = set(tf.all_variables())

        # EXCEPT model variables (that has been marked as trainable)
        model_variables = set(tf.trainable_variables())
        init_op = tf.initialize_variables(list(all_variables -
                                               model_variables))

        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(init_op)

            # Start input enqueue threads.
            print("Starting input enqueue threads. Please wait...")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                processed = 0
                sum_accuracy = 0.0
                #sum_accuracy_per_class = {label: 0.0
                #                          for label in PASCAL2PASCIFAR.values()
                #                          }
                while not coord.should_stop():
                    image_batch, label_batch = sess.run(
                        [image_queue, label_queue])

                    # run prediction on images resized
                    predicted_labels, batch_accuracy = sess.run(
                        [predictions, accuracy],
                        feed_dict={
                            "images_:0": image_batch,
                            labels_: label_batch,
                        })

                    print(label_batch)
                    print(predicted_labels)
                    print(batch_accuracy)

                    sum_accuracy += batch_accuracy
                    processed += 1

            except tf.errors.OutOfRangeError:
                print("[I] Done. Test completed!")
                print("Processed {} images".format(processed * BATCH_SIZE))
                print("Avg accuracy: {}".format(sum_accuracy / processed))
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
