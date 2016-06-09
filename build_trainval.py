#Copyright (C) 2016 Paolo Galeone <nessuno@nerdz.eu>
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""./build_trainval.py PASCAL_2012/VOCdevkit/VOC2012 cropped_dataset
Extracts the 20 categories from the PASCAL dataset (argv[1]).
Crop every image to annotated bounding boxes as described in argv[1]/Annotations/
Creates the cropped_dataset/ts.csv file. Splits the ts.csv file in train.csv & validation.csv.
Outputs the average input width and height.
"""

import glob
import xml.etree.ElementTree as etree
import sys
import os
import csv
from collections import defaultdict
import math
import cv2

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

FIELD_NAMES = ["file", "width", "height", "label"]

VALIDATION_DATASET_PERCENTACE = 1 / 90


def crop(file_name, rect):
    """Read file_name, extracts the ROI using rect[y1,y2,x1,x2]"""
    image = cv2.imread(file_name)
    return image[rect[0]:rect[1], rect[2]:rect[3]]


def split_dataset(base_path):
    """Split the out_path/ts.csv file into train and test csv files.
    Creates validation.csv and train.csv in the current directory.
    Splits accoring to the VALIDATION_DATASET_PERCENTACE"""
    # create a dictionary of list
    # every list contains the rows for the specified label
    print("Splitting dataset.\nValidation percentage: {}".format(
        VALIDATION_DATASET_PERCENTACE))
    labels = defaultdict(list)
    tot_line = 0
    with open(base_path + "/ts.csv", 'r') as ts_file:
        reader = csv.DictReader(ts_file)
        for row in reader:
            label = row["label"]
            labels[label].append(row)
            tot_line += 1

    print("Numer of examples: {}".format(tot_line))
    current_dir = os.path.abspath(os.getcwd())
    print("Creating train.csv, validation.csv in {}".format(current_dir))

    train_file = open("{}/train.csv".format(current_dir), "w")
    validation_file = open("{}/validation.csv".format(current_dir), "w")

    tf_writer = csv.DictWriter(train_file, FIELD_NAMES)
    vf_writer = csv.DictWriter(validation_file, FIELD_NAMES)

    tf_writer.writeheader()
    vf_writer.writeheader()

    tot_validation = 0

    for label in labels:
        items_count = len(labels[label])
        validation_count = math.floor(items_count *
                                      VALIDATION_DATASET_PERCENTACE)
        train_count = items_count - validation_count

        tot_validation += validation_count

        print("Label: {}\n\tItems:{}\n\tValidation: {}\n\tTrain: {}".format(
            label, items_count, validation_count, train_count))

        while validation_count > 0:
            vf_writer.writerow(labels[label].pop())
            validation_count -= 1

        while train_count > 0:
            tf_writer.writerow(labels[label].pop())
            train_count -= 1

    train_file.close()
    validation_file.close()
    print(
        "Number of validation examples: {}\nNumber of training examples: {}".format(
            tot_validation, tot_line - tot_validation))
    return 0


def main(argv):
    """ main """
    len_argv = len(argv)
    if len_argv not in (2, 3):
        print(
            "usage: pascal_cropper.py /path/of/VOC<year> /path/of/cropped_dataset/",
            file=sys.stderr)
        return 1

    if not os.path.exists(argv[0]):
        print("{} does not exists".format(argv[0]))
        return 1

    if not os.path.exists(argv[1]):
        os.makedirs(argv[1])

    # avg_shape will contain the average height, width of the extracted ROIs
    # 3 is the number of channels
    # avg_shape[width, height, 3]
    avg_shape = [0, 0, 3]

    out_path = os.path.abspath(argv[1])
    current_dir = os.path.abspath(os.getcwd())

    ts_csv_abs = "{}/ts.csv".format(out_path)

    if os.path.exists(ts_csv_abs) and (
            not os.path.exists("{}/train.csv".format(current_dir)) or
            not os.path.exists("{}/validation.csv".format(current_dir))):
        return split_dataset(out_path)

    if os.path.exists(ts_csv_abs):
        print("Dataset already exists. Remove {} to rebuild it".format(
            ts_csv_abs))
        return 0

    i = 0
    with open(ts_csv_abs, mode='w') as csv_file:
        # header
        writer = csv.DictWriter(csv_file, FIELD_NAMES)
        writer.writeheader()

        images_path = "{}/JPEGImages/".format(argv[0])

        for current_class in CLASSES:
            lines = open("{}/ImageSets/Main/{}_trainval.txt".format(argv[
                0], current_class)).read().strip().split("\n")

            for line in lines:
                splitted = line.split()
                if len(splitted) < 1:
                    print(splitted, line, current_class)
                if splitted[1] == "-1":
                    continue
                image_xml = "{}/Annotations/{}.xml".format(argv[0],
                                                           splitted[0])
                image_file = "{}.jpg".format(splitted[0])

                tree = etree.parse(image_xml)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                for obj in root.iter('object'):
                    # skip difficult & object.name not in current class
                    label = obj.find('name').text

                    if label != current_class:
                        continue

                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        continue

                    bndbox = obj.find('bndbox')
                    rect = [0, 0, 0, 0]
                    #y1
                    rect[0] = int(float(bndbox.find('ymin').text))
                    #y2
                    rect[1] = int(float(bndbox.find('ymax').text))
                    #x1
                    rect[2] = int(float(bndbox.find('xmin').text))
                    #x2
                    rect[3] = int(float(bndbox.find('xmax').text))

                    roi = crop(images_path + image_file, rect)
                    width, height = roi.shape[1], roi.shape[0]
                    avg_shape[0] += width
                    avg_shape[1] += height

                    # use numeric id for label in csv
                    label_id = CLASSES.index(current_class)

                    # check if the the image file name (witout suffix)
                    # is already present in the destionation folder
                    # this means that in the same original image we got multiple classes.
                    look_for = "{}/{}*.jpg".format(out_path, splitted[0])
                    in_folder = len(glob.glob(look_for))
                    new_image_file = "{}_{}.jpg".format(splitted[0], in_folder)

                    # save file and append row to csv
                    cv2.imwrite(out_path + "/" + new_image_file, roi)
                    writer.writerow({"file": new_image_file,
                                     "width": width,
                                     "height": height,
                                     "label": label_id})
                    i += 1

    print("Average width & height")
    print(int(float(avg_shape[0] / i)), int(float(avg_shape[1] / i)))

    return split_dataset(out_path)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
