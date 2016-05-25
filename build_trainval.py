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


def crop(file_name, rect):
    """Read file_name, extracts the ROI using rect[y1,y2,x1,x2]"""
    image = cv2.imread(file_name)
    return image[rect[0]:rect[1], rect[2]:rect[3]]


def split_dataset(base_path):
    """Split the out_path/ts.csv file into train and test csv files
    The distribution is 2/3 train, 1/3 validation for every class."""
    # create a dictionary of list
    # every list contains the rows for the specified label
    print("Splitting dataset...")
    labels = defaultdict(list)
    tot_line = 0
    with open(base_path + "/ts.csv", 'r') as ts_file:
        reader = csv.DictReader(ts_file)
        for row in reader:
            label = row["label"]
            labels[label].append(row)
            tot_line += 1

    print(tot_line)

    train_file = open(base_path + "/train.csv", "w")
    validation_file = open(base_path + "/validation.csv", "w")

    tf_writer = csv.DictWriter(train_file, FIELD_NAMES)
    vf_writer = csv.DictWriter(validation_file, FIELD_NAMES)

    tf_writer.writeheader()
    vf_writer.writeheader()

    for label in labels:
        items_count = len(labels[label])
        validation_count = math.floor(items_count / 3)
        train_count = items_count - validation_count

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
    return 0


def main(argv):
    """ main """
    len_argv = len(argv)
    if len_argv not in (2, 3):
        print("usage: pascal_cropper.py /path/of/VOC<year> /path/of/output",
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

    if os.path.exists(out_path + "/ts.csv") and (
            not os.path.exists(out_path + "/train.csv") or
            not os.path.exists(out_path + "/validation.csv")):
        return split_dataset(out_path)

    if os.path.exists(out_path + "/ts.csv"):
        print("Dataset already created. Remove {}/ts.csv to rebuild it".format(
            out_path))
        return 0

    i = 0
    with open(out_path + '/ts.csv', mode='w') as csv_file:
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
