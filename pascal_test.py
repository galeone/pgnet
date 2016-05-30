"""./pascal_test.py PASCAL_2012_test_dataset/VOCdevkit/VOC2012

Reads the file list in: argv[1]/ImageSets/Main/test.txt
Creates in the current directory a subfolder for the results (results/).
For every of the 20 classes, creates a file with the following format:
    filename: comp1_cls_test_<class>.txt:
    <filename (no extension)> confidence // eg: 2009_000001 0.056313
"""

import argparse
import os
import sys
import tensorflow as tf
import build_trainval
import pgnet
import train


def main(args):
    """ main """

    if not os.path.exists(args.test_ds):
        print("{} does not exists".format(args.test_ds))
        return 1

    # export model.pb from session dir. Skip if model.pb already exists
    train.export_model()

    current_dir = os.path.abspath(os.getcwd())
    results_dir = "{}/results".format(current_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # open the test.txt file and extract the content (files to test)
    lines = open("{}/ImageSets/Main/test.txt".format(args.test_ds)).read(
    ).strip().split("\n")

    # open a file for every class
    files = {label:
             open(results_dir + "/comp1_cls_test_{}.txt".format(label), "w")
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

        base_path = "{}/JPEGImages/".format(args.test_ds)
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            print("Processing {} files".format(len(lines)))
            for idx, image_line in enumerate(lines):
                # prepend the path
                image_path = tf.constant("{}{}.jpg".format(base_path,
                                                           image_line))
                # read the image
                image = tf.image.decode_jpeg(tf.read_file(image_path))

                # subtract off the mean and divide by the variance of the pixels
                image = tf.image.per_image_whitening(image)

                # crop/reshape image for evaluation
                # Bug in tf not solved yet:
                # https://github.com/tensorflow/tensorflow/issues/521
                # we can't use resize_image_with_crop_or_pad
                #image = tf.image.resize_image_with_crop_or_pad(
                #    image, pgnet.INPUT_SIDE, pgnet.INPUT_SIDE)

                # Instead we use resize_nearest_neigbor
                # pgnet accepts a batch of images as input, add the "batch" dimension.
                # in addiction, resize_nearest_neigbor accepts batch of images only
                image = tf.expand_dims(image, 0)

                image = tf.image.resize_nearest_neighbor(
                    image, [pgnet.INPUT_SIDE, pgnet.INPUT_SIDE])
                # feed the input placeholder _images with the current "batch" (1) of image
                print("{}: {}".format(idx, image_line))
                image_evaluated = image.eval()
                predictions_prob = sess.run(softmax,
                                            feed_dict={
                                                "keep_prob_:0": 1.0,
                                                "images_:0": image_evaluated,
                                            })

                # remove batch size (we're processing one image at time)
                predictions_prob = predictions_prob[0]
                for idx, pred in enumerate(predictions_prob):
                    files[build_trainval.CLASSES[idx]].write("{} {}\n".format(
                        image_line, pred))

        # close all files
        for label in files:
            files[label].close()


if __name__ == "__main__":
    # pylint: disable=C0103
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--device", default="/gpu:1")
    parser.add_argument("--test-ds")
    sys.exit(main(parser.parse_args()))
