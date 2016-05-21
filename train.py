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

# train parameters
MAX_ITERATIONS = 10**100 + 1
DISPLAY_STEP = 1
VALIDATION_STEP = 100
MIN_VALIDATION_ACCURACY = 0.9


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
                train_images, train_labels = pascal_input.train_inputs(
                    CSV_PATH, pgnet.BATCH_SIZE)

            with tf.variable_scope("validation_input"):
                validation_images, validation_labels = pascal_input.validation_inputs(
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

            with tf.variable_scope("validation_accuracy"):
                # reshape logits to a [-1, NUM_CLASS] vector
                # (remeber that pgnet is fully convolutional)
                reshaped_logits = tf.reshape(logits, [-1, pgnet.NUM_CLASS])

                # returns the label predicted
                # reshaped_logits contains NUM_CLASS values in NUM_CLASS
                # positions. Each value is the probability for the position class.
                # Returns the index (thus the label) with highest probability, for each line
                # [-1] vector
                predictions = tf.argmax(reshaped_logits, 1)

                # correct predictions
                # [-1] vector
                correct_predictions = tf.equal(validation_labels, predictions)
                # validation_accuracy per batch
                validation_accuracy_per_batch = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32),
                    name="validation_accuracy_per_batch")

                # define a placeholder for the average validation_accuracy over the validation dataset
                average_validation_accuracy_ = tf.placeholder(
                    tf.float32, name="avg_validation_accuracy")
                # attach a summary to the placeholder
                tf.scalar_summary("avg_validation_accuracy",
                                  average_validation_accuracy_)

                # create a saver: to store current computation and restore the graph
                # useful when the train step has been interrupeted
            saver = tf.train.Saver(tf.all_variables())

            # collect summaries
            summary_op = tf.merge_all_summaries()

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
                validation_accuracy = 0.0
                for step in range(MAX_ITERATIONS):
                    # get train inputs
                    images, labels = sess.run([train_images, train_labels])
                    start = time.time()
                    #train
                    _, loss_val = sess.run(
                        [train_op, loss_op],
                        feed_dict={
                            keep_prob_: 0.5,
                            images_: images,
                            labels_: labels,
                        })

                    duration = time.time() - start
                    stop_train = False
                    if np.isnan(loss_val):
                        print('Model diverged with loss = NaN',
                              file=sys.stderr)
                        # print reshaped logits value for debug purposes
                        print(sess.run(reshaped_logits,
                                       feed_dict={
                                           keep_prob_: 1.,
                                           images_: images,
                                           labels_: labels
                                       }))
                        return 1

                    if step % VALIDATION_STEP == 0 and step > 0:
                        num_validation_batches = int(
                            pascal_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL /
                            pgnet.BATCH_SIZE)
                        validation_accuracy_sum = 0.0
                        # early stop: if the accuracy batch is less then the MIN_VALIDATION_ACCURACY/2
                        # the probability that the global accuracy is >= MIN_VALIDATION_ACCURACY is low
                        used_batches = 0
                        for i in range(num_validation_batches):
                            # get validation inputs
                            # do not override images,labels variable that are required in summeries
                            imgs, lbls = sess.run([validation_images,
                                                   validation_labels])
                            batch_validation_accuracy = sess.run(
                                validation_accuracy_per_batch,
                                feed_dict={
                                    images_: imgs,
                                    labels_: lbls,
                                    keep_prob_: 1.0
                                })

                            used_batches += 1
                            print("Validation batch {}, accuracy: {}".format(
                                i + 1, batch_validation_accuracy))
                            validation_accuracy_sum += batch_validation_accuracy
                            if batch_validation_accuracy <= MIN_VALIDATION_ACCURACY / 2:
                                break

                        validation_accuracy = validation_accuracy_sum / used_batches
                        # save validation_accuracy in summary (on next display step) stop_train
                        if validation_accuracy > MIN_VALIDATION_ACCURACY:
                            stop_train = True

                    if step % DISPLAY_STEP == 0 and step > 0:
                        num_examples_per_step = pgnet.BATCH_SIZE
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        print(
                            "{} step: {} loss: {} ({} examples/sec; {} batch/sec)".format(
                                datetime.now(), step, loss_val,
                                examples_per_sec, sec_per_batch))

                        # create summary for this train step
                        summary_line = sess.run(
                            summary_op,
                            feed_dict={keep_prob_: 0.5,
                                       average_validation_accuracy_:
                                       validation_accuracy,
                                       images_: images,
                                       labels_: labels})

                        # global_step in add_summary is the local step (thank you tensorflow)
                        summary_writer.add_summary(summary_line,
                                                   global_step=step)

                        # save the current session (until this step) in the session dir
                        # export a checkpint in the format SESSION_DIR/model-<global_step>.meta
                        # always pass 0 to global step in order to have only one file in the folder
                        saver.save(sess, SESSION_DIR + "/model", global_step=0)
                        if stop_train:
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
