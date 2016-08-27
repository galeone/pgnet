#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""train the model"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import pgnet
import pascal_input

# graph parameteres
SESSION_DIR = "session"
SUMMARY_DIR = "summary"
TRAINED_MODEL_FILENAME = "model.pb"

# cropped pascal parameters
CSV_PATH = "~/data/PASCAL_2012_cropped"

# train & validation parameters
DISPLAY_STEP = 5
MEASUREMENT_STEP = 20
STEP_FOR_EPOCH = math.ceil(pascal_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                           pgnet.BATCH_SIZE)
MAX_ITERATIONS = STEP_FOR_EPOCH * 500

# stop when
AVG_VALIDATION_ACCURACY_EPOCHS = 50
# list of average validation at the end of every epoch
AVG_VALIDATION_ACCURACIES = [0.0
                             for _ in range(AVG_VALIDATION_ACCURACY_EPOCHS)]

# tensorflow saver constant
SAVE_MODEL_STEP = math.ceil(STEP_FOR_EPOCH / 2)


def train(args):
    """train model"""

    if not os.path.exists(SESSION_DIR):
        os.makedirs(SESSION_DIR)
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Number of classes in the dataset plus 1.
    # Labelp pascal_input. NUM_CLASSES + 1 is reserved for
    # an (unused) background class.
    num_classes = pascal_input.NUM_CLASSES + 1

    # if the trained model does not exist
    if not os.path.exists(TRAINED_MODEL_FILENAME):
        # train graph is the graph that contains the variable
        graph = tf.Graph()

        # create a scope for the graph. Place operations on cpu:0
        # if not otherwise specified
        with graph.as_default(), tf.device('/cpu:0'):

            with tf.variable_scope("train_input"):
                # get the train input
                train_images_queue, train_labels_queue = pascal_input.train(
                    CSV_PATH, pgnet.BATCH_SIZE)

            with tf.variable_scope("validation_input"):
                validation_images_queue, validation_labels_queue = pascal_input.validation(
                    CSV_PATH, pgnet.BATCH_SIZE)

            with tf.device(args.device):  #GPU
                # train global step
                global_step = tf.Variable(
                    0, trainable=False, name="global_step")

                # model inputs, used in train and validation
                labels_ = tf.placeholder(
                    tf.int64, shape=[None], name="labels_")

                is_training_, keep_prob_, images_, logits = pgnet.define_model(
                    num_classes, train_phase=True)

                # loss op
                loss_op = pgnet.loss(logits, labels_)

                # train op
                train_op = pgnet.train(loss_op, global_step)

            # collect summaries for the previous defined variables
            summary_op = tf.merge_all_summaries()

            with tf.variable_scope("accuracy"):
                # since pgnet if fully convolutional remove dimensions of size 1
                reshaped_logits = tf.squeeze(logits, [1, 2])

                # returns the label predicted
                # reshaped_logits contains num_classes values in num_classes
                # positions. Each value is the probability for the position class.
                # Returns the index (thus the label) with highest probability, for each line
                # [BATCH_SIZE] vector
                predictions = tf.argmax(reshaped_logits, 1)

                # correct predictions
                # [BATCH_SIZE] vector
                correct_predictions = tf.equal(labels_, predictions)

                accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, tf.float32), name="accuracy")

                # use a separate summary op for the accuracy (that's shared between test
                # and validation)

                # change only the content of the placeholder that names the summary
                accuracy_name_ = tf.placeholder(tf.string, [])

                # attach a summary to the placeholder
                accuracy_summary_op = tf.scalar_summary(accuracy_name_,
                                                        accuracy)

            # create a saver: to store current computation and restore the graph
            # useful when the train step has been interrupeted
            variables_to_save = pgnet.variables_to_save([global_step])
            saver = tf.train.Saver(variables_to_save)

            # tensor flow operator to initialize all the variables in a session
            init_op = tf.initialize_all_variables()

            with tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:

                def validate():
                    """get validation inputs and run validation.
                    Returns:
                        validation_accuracy, summary_line
                    """
                    # get validation inputs
                    validation_images, validation_labels = sess.run(
                        [validation_images_queue, validation_labels_queue])

                    validation_accuracy, summary_line = sess.run(
                        [accuracy, accuracy_summary_op],
                        feed_dict={
                            images_: validation_images,
                            labels_: validation_labels,
                            keep_prob_: 1.0,
                            is_training_: False,
                            accuracy_name_: "validation_accuracy"
                        })
                    return validation_accuracy, summary_line

                # initialize variables
                sess.run(init_op)

                # Start the queue runners (input threads)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # restore previous session if exists
                checkpoint = tf.train.get_checkpoint_state(SESSION_DIR)
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                else:
                    print("[I] Unable to restore from checkpoint")

                summary_writer = tf.train.SummaryWriter(
                    SUMMARY_DIR + "/train", graph=sess.graph)

                total_start = time.time()
                current_epoch = 0
                sum_validation_accuracy = 0
                for step in range(MAX_ITERATIONS):
                    # get train inputs
                    train_images, train_labels = sess.run([train_images_queue,
                                                           train_labels_queue])

                    start = time.time()
                    # train, get loss value, get summaries
                    _, loss_val, summary_line, gs_value = sess.run(
                        [train_op, loss_op, summary_op, global_step],
                        feed_dict={
                            keep_prob_: 0.5,
                            is_training_: True,
                            images_: train_images,
                            labels_: train_labels,
                        })
                    duration = time.time() - start

                    # save summary for current step
                    summary_writer.add_summary(
                        summary_line, global_step=gs_value)

                    if np.isnan(loss_val):
                        print(
                            'Model diverged with loss = NaN', file=sys.stderr)
                        # print reshaped logits value for debug purposes
                        print(
                            sess.run(reshaped_logits,
                                     feed_dict={
                                         keep_prob_: 1.0,
                                         is_training_: False,
                                         images_: train_images,
                                         labels_: train_labels
                                     }),
                            file=sys.stderr)
                        return 1

                    if step % DISPLAY_STEP == 0 and step > 0:
                        examples_per_sec = pgnet.BATCH_SIZE / duration
                        sec_per_batch = float(duration)
                        print(
                            "{} step: {} loss: {} ({} examples/sec; {} batch/sec)".
                            format(datetime.now(), gs_value, loss_val,
                                   examples_per_sec, sec_per_batch))

                    stop_training = False
                    if step % MEASUREMENT_STEP == 0 and step > 0:
                        validation_accuracy, summary_line = validate()
                        # save summary for validation_accuracy
                        summary_writer.add_summary(
                            summary_line, global_step=gs_value)

                        # test accuracy
                        test_accuracy, summary_line = sess.run(
                            [accuracy, accuracy_summary_op],
                            feed_dict={
                                images_: train_images,
                                labels_: train_labels,
                                keep_prob_: 1.0,
                                is_training_: False,
                                accuracy_name_: "training_accuracy"
                            })

                        # save summary for training accuracy
                        summary_writer.add_summary(
                            summary_line, global_step=gs_value)

                        print(
                            "{} step: {} validation accuracy: {} training accuracy: {}".
                            format(datetime.now(), gs_value,
                                   validation_accuracy, test_accuracy))

                        sum_validation_accuracy += validation_accuracy

                    if step % STEP_FOR_EPOCH == 0 and step > 0:
                        # current validation accuracy
                        current_validation_accuracy = sum_validation_accuracy * MEASUREMENT_STEP / STEP_FOR_EPOCH
                        print(
                            "Epoch {} finised. Average validation accuracy/epoch: {}".
                            format(current_epoch, current_validation_accuracy))

                        # sum previous avg accuracy
                        history_avg_accuracy = sum(
                            AVG_VALIDATION_ACCURACIES) / AVG_VALIDATION_ACCURACY_EPOCHS

                        # if avg accuracy is not increased, after
                        # AVG_VALIDATION_ACCURACY_NOT_INCREASED_AFTER_EPOCH, exit
                        if current_validation_accuracy <= history_avg_accuracy:
                            print(
                                "Average validation accuracy not increased after {} epochs. Exit".
                                format(AVG_VALIDATION_ACCURACY_EPOCHS))
                            # exit using stop_training flag, in order to save current status
                            stop_training = True

                        # save avg validation accuracy in the next slot
                        AVG_VALIDATION_ACCURACIES[
                            current_epoch %
                            AVG_VALIDATION_ACCURACY_EPOCHS] = current_validation_accuracy

                        current_epoch += 1
                        sum_validation_accuracy = 0

                    if step % SAVE_MODEL_STEP == 0 or (
                            step + 1) == MAX_ITERATIONS or stop_training:
                        # save the current session (until this step) in the session dir
                        # export a checkpint in the format SESSION_DIR/model-<global_step>.meta
                        # always pass 0 to global step in order to have only one file in the folder
                        saver.save(sess, SESSION_DIR + "/model", global_step=0)

                    if stop_training:
                        break

                # end of train
                print("Train completed in {}".format(time.time() -
                                                     total_start))

                # save train summaries to disk
                summary_writer.flush()

                # When done, ask the threads to stop.
                coord.request_stop()
                # Wait for threads to finish.
                coord.join(threads)

        # if here, the summary dir contains the trained model
        current_dir = os.path.abspath(os.getcwd())
        pgnet.export_model(num_classes, current_dir + "/session", "model-0",
                           "model.pb")

    else:
        print("Trained model {} already exits".format(TRAINED_MODEL_FILENAME))
    return 0


if __name__ == "__main__":
    ARG_PARSER = argparse.ArgumentParser(description="Train the model")
    ARG_PARSER.add_argument("--device", default="/gpu:1")
    sys.exit(train(ARG_PARSER.parse_args()))
