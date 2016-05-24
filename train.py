#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""train the model"""

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import freeze_graph
import pgnet
import pascal_input

# graph parameteres
SESSION_DIR = "session"
SUMMARY_DIR = "summary"
TRAINED_MODEL_FILENAME = "model.pb"

# cropped pascal parameters
CSV_PATH = "~/data/PASCAL_2012_cropped"

# train & validation parameters
MAX_ITERATIONS = 10**100 + 1
DISPLAY_STEP = 1
MEASUREMENT_STEP = 10
MIN_VALIDATION_ACCURACY = 0.9
NUM_VALIDATION_BATCHES = int(pascal_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL /
                             pgnet.BATCH_SIZE)

# tensorflow saver constant
SAVE_MODEL_STEP = 1000


def train(args):
    """train model"""

    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # if the trained model does not exist
    if not os.path.exists(TRAINED_MODEL_FILENAME):
        # train graph is the graph that contains the variable
        graph = tf.Graph()

        # create a scope for the graph. Place operations on cpu:0
        # if not otherwise specified
        with graph.as_default(), tf.device('/cpu:0'):
            # model dropout keep_prob placeholder
            keep_prob_ = tf.placeholder(tf.float32, name="keep_prob")

            # train global step
            global_step = tf.Variable(0, trainable=False)

            with tf.variable_scope("train_input"):
                # get the train input
                train_images_queue, train_labels_queue = pascal_input.train_inputs(
                    CSV_PATH, pgnet.BATCH_SIZE)

            with tf.variable_scope("validation_input"):
                validation_images_queue, validation_labels_queue = pascal_input.validation_inputs(
                    CSV_PATH, pgnet.BATCH_SIZE)

            with tf.device(args.device):  #GPU
                # model inputs, used in train and validation
                labels_ = tf.placeholder(tf.int64,
                                         shape=[None],
                                         name="labels_")
                images_ = tf.placeholder(tf.float32,
                                         shape=[None, pgnet.INPUT_SIDE,
                                                pgnet.INPUT_SIDE,
                                                pgnet.INPUT_DEPTH],
                                         name="images_")

                # build a graph that computes the logits predictions from the model
                logits = pgnet.get(images_, keep_prob_)

                # loss op
                loss_op = pgnet.loss(logits, labels_)

                # train op
                train_op = pgnet.train(loss_op, global_step)

            # collect summaries for the previous defined variables
            summary_op = tf.merge_all_summaries()

            with tf.variable_scope("accuracy"):
                # reshape logits to a [-1, NUM_CLASS] vector
                # (remeber that pgnet is fully convolutional)
                reshaped_logits = tf.reshape(logits, [-1, pgnet.NUM_CLASS])

                # returns the label predicted
                # reshaped_logits contains NUM_CLASS values in NUM_CLASS
                # positions. Each value is the probability for the position class.
                # Returns the index (thus the label) with highest probability, for each line
                # [BATCH_SIZE] vector
                predictions = tf.argmax(reshaped_logits, 1)

                # correct predictions
                # [BATCH_SIZE] vector
                correct_predictions = tf.equal(labels_, predictions)

                accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32),
                    name="accuracy")

                # use a separate summary op for the accuracy (that's shared between test
                # and validation)

                # change only the content of the placeholder that names the summary
                accuracy_name_ = tf.placeholder(tf.string, [])

                # attach a summary to the placeholder
                accuracy_summary_op = tf.scalar_summary(accuracy_name_,
                                                        accuracy)

            # create a saver: to store current computation and restore the graph
            # useful when the train step has been interrupeted
            saver = tf.train.Saver(tf.all_variables())

            # tensor flow operator to initialize all the variables in a session
            init_op = tf.initialize_all_variables()

            with tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                # initialize variables
                sess.run(init_op)

                # Start the queue runners (input threads)
                tf.train.start_queue_runners(sess=sess)

                # restore previous session if exists
                checkpoint = tf.train.get_checkpoint_state(SESSION_DIR)
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                else:
                    print("[I] Unable to restore from checkpoint")

                summary_writer = tf.train.SummaryWriter(SUMMARY_DIR + "/train",
                                                        graph=sess.graph)

                total_start = time.time()
                for step in range(MAX_ITERATIONS):
                    # get train inputs
                    train_images, train_labels = sess.run([train_images_queue,
                                                           train_labels_queue])

                    start = time.time()
                    # train, get loss value, get summaries
                    _, loss_val, summary_line = sess.run(
                        [train_op, loss_op, summary_op],
                        feed_dict={
                            keep_prob_: 0.5,
                            images_: train_images,
                            labels_: train_labels,
                        })
                    duration = time.time() - start

                    # save summary for current step
                    summary_writer.add_summary(summary_line, global_step=step)

                    if np.isnan(loss_val):
                        print('Model diverged with loss = NaN',
                              file=sys.stderr)
                        # print reshaped logits value for debug purposes
                        print(
                            sess.run(reshaped_logits,
                                     feed_dict={
                                         keep_prob_: 1.0,
                                         images_: train_images,
                                         labels_: train_labels
                                     }),
                            file=sys.stderr)
                        return 1

                    if step % DISPLAY_STEP == 0 and step > 0:
                        examples_per_sec = pgnet.BATCH_SIZE / duration
                        sec_per_batch = float(duration)
                        print(
                            "{} step: {} loss: {} ({} examples/sec; {} batch/sec)".format(
                                datetime.now(), step, loss_val,
                                examples_per_sec, sec_per_batch))

                    validation_accuracy_reached = False
                    if step % MEASUREMENT_STEP == 0 and step > 0:
                        # get validation inputs
                        validation_images, validation_labels = sess.run(
                            [validation_images_queue, validation_labels_queue])

                        validation_accuracy, summary_line = sess.run(
                            [accuracy, accuracy_summary_op],
                            feed_dict={
                                images_: validation_images,
                                labels_: validation_labels,
                                keep_prob_: 1.0,
                                accuracy_name_: "validation_accuracy"
                            })

                        # save summary for validation_accuracy
                        summary_writer.add_summary(summary_line,
                                                   global_step=step)

                        if validation_accuracy > MIN_VALIDATION_ACCURACY:
                            validation_accuracy_reached = True

                        test_accuracy, summary_line = sess.run(
                            [accuracy, accuracy_summary_op],
                            feed_dict={
                                images_: train_images,
                                labels_: train_labels,
                                keep_prob_: 1.0,
                                accuracy_name_: "training_accuracy"
                            })
                        # save summary for training accuracy
                        summary_writer.add_summary(summary_line,
                                                   global_step=step)

                        print(
                            "{} step: {} validation accuracy: {} training accuracy: {}".format(
                                datetime.now(
                                ), step, validation_accuracy, test_accuracy))

                    if step % SAVE_MODEL_STEP == 0 or (
                            step + 1
                    ) == MAX_ITERATIONS or validation_accuracy_reached:
                        # save the current session (until this step) in the session dir
                        # export a checkpint in the format SESSION_DIR/model-<global_step>.meta
                        # always pass 0 to global step in order to have only one file in the folder
                        saver.save(sess, SESSION_DIR + "/model", global_step=0)

                    if validation_accuracy_reached:
                        break

                # end of train
                print("Train completed in {}".format(time.time() -
                                                     total_start))

                # save train summaries to disk
                summary_writer.flush()

                # save model skeleton (the empty graph, its definition)
                tf.train.write_graph(graph.as_graph_def(),
                                     SESSION_DIR,
                                     "skeleton.pb",
                                     as_text=False)

                freeze_graph.freeze_graph(SESSION_DIR + "/skeleton.pb", "",
                                          True, SESSION_DIR + "/model-0",
                                          "softmax_linear/out",
                                          "save/restore_all", "save/Const:0",
                                          TRAINED_MODEL_FILENAME, False, "")
    else:
        print("Trained model {} already exits".format(TRAINED_MODEL_FILENAME))
    return 0


if __name__ == "__main__":
    # pylint: disable=C0103
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--device", default="/gpu:1")
    sys.exit(train(parser.parse_args()))
