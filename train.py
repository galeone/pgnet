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
MIN_ACCURACY = 0.9
MAX_ITERATIONS = 10**100 + 1
DISPLAY_STEP = 100


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

            with tf.device(args.device):
                # get the train input
                images, labels = pascal_input.train_inputs(CSV_PATH,
                                                           pgnet.BATCH_SIZE)

                # build a graph that computes the logits predictions from the model
                logits = pgnet.get(images, keep_prob_)

                # loss op
                loss_op = pgnet.loss(logits, labels)

                # train op
                train_op = pgnet.train(loss_op, global_step)

            # collect summaries
            summary_op = tf.merge_all_summaries()

            # tensor flow operator to initialize all the variables in a session
            init_op = tf.initialize_all_variables()

            with tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                # initialize variables
                sess.run(init_op)

                # Start the queue runners (input threads)
                tf.train.start_queue_runners(sess)

                # create a saver: to store current computation and restore the graph
                # useful when the train step has been interrupeted
                saver = tf.train.Saver(tf.all_variables())

                # restore previous session if exists
                checkpoint = tf.train.get_checkpoint_state(SESSION_DIR)
                if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                else:
                    print("[I] Unable to restore from checkpoint")

                summary_writer = tf.train.SummaryWriter(SUMMARY_DIR + "/train",
                                                        graph=sess.graph)

                for step in range(MAX_ITERATIONS):
                    start = time.time()
                    #train
                    _, loss_val, gs_value = sess.run(
                        [train_op, loss_op, global_step],
                        feed_dict={
                            keep_prob_: 0.5
                        })

                    duration = time.time() - start.time()
                    assert not np.isnan(
                        loss_val), 'Model diverged with loss = NaN'

                    if step % DISPLAY_STEP == 0 and step > 0:
                        # we don't need accuracy on a validation set, during training
                        # because we have the loss: loss decrease & accuracy increase

                        num_examples_per_step = pgnet.BATCH_SIZE
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        print(
                            "{} step: {} loss: {} ({} examples/sec; {} batch/sec)".format(
                                datetime.now(), gs_value, loss_val,
                                examples_per_sec, sec_per_batch))

                        # create summary for this train step
                        summary_line = sess.run(summary_op,
                                                feed_dict={
                                                    keep_prob_: 0.5
                                                })
                        # global_step in add_summary is the local step (thank you tensorflow)
                        summary_writer.add_summary(summary_line,
                                                   global_step=step)

                        # save the current session (until this step) in the session dir
                        # export a checkpint in the format SESSION_DIR/model-<global_step>.meta
                        # always pass 0 to global step in order to
                        # have only one file in the folder
                        saver.save(sess, SESSION_DIR + "/model", global_step=0)

                # end of train

                # save train summaries to disk
                summary_writer.flush()

                # save model skeleton (the empty graph, its definition)
                tf.train.write_graph(graph.as_graph_def(),
                                     SESSION_DIR,
                                     "skeleton.pb",
                                     as_text=False)

                freeze_graph.freeze_graph(SESSION_DIR + "/skeleton.pb", "",
                                          True, SESSION_DIR + "/model-0",
                                          "softmax_linear/softmax_linear:0",
                                          "save/restore_all", "save/Const:0",
                                          TRAINED_MODEL_FILENAME, False, "")
    else:
        print("Trained model %s already exits" % (TRAINED_MODEL_FILENAME))
    return 0


if __name__ == "__main__":
    # pylint: disable=C0103
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--device", default="/gpu:1")
    sys.exit(train(parser.parse_args()))
