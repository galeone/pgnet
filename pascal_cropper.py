"""./pascal_cropper.py PASCAL_2012/VOCdevkit/VOC2012 cropped_dataset train=True
Extracts the 20 categories from the PASCAL dataset (argv[1]).
Crop every image to annotated bounding boxes as described in argv[1]/Annotations/
Creates the cropped_dataset/ts.csv file.
If train (argv[3]) is present, splits the ts.csv file in train.csv and validation.csv.
Outputs the average input widht and height.
"""
import xml.etree.ElementTree as etree
import sys
import os
import csv
from collections import defaultdict
import math
import cv2

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def crop(file_name, rect):
    """Read file_name, extracts the ROI using rect[y1,y2,x1,x2]"""

    image = cv2.imread(file_name)
    return image[rect[0]:rect[1], rect[2]:rect[3]]


# https://stackoverflow.com/questions/431684/how-do-i-cd-in-python/13197763#13197763
class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


field_names = ["file", "width", "height", "label"]


def split_dataset(base_path):
    """Split the out_path/ts.csv file into train and test csv files
    The distribution is 2/3 train, 1/3 validation for every class."""
    # create a dictionary of list
    # every list contains the rows for the specified label
    print("Splitting dataset...")
    labels = defaultdict(list)
    with open(base_path + "/ts.csv", 'r') as ts_file:
        reader = csv.DictReader(ts_file)
        for row in reader:
            label = row["label"]
            labels[label].append(row)

    train_file = open(base_path + "/train.csv", "w")
    validation_file = open(base_path + "/validation.csv", "w")

    tf_writer = csv.DictWriter(train_file, field_names)
    vf_writer = csv.DictWriter(validation_file, field_names)

    tf_writer.writeheader()
    vf_writer.writeheader()

    for label in labels:
        items_count = len(labels[label])
        validation_count = math.ceil(items_count / 3)

        print("Label: {}\n\tValidation: {}\n\tTrain: {}".format(
            label, validation_count, items_count - validation_count))

        while validation_count >= 0:
            vf_writer.writerow(labels[label].pop())
            validation_count -= 1

        while labels[label] != []:
            tf_writer.writerow(labels[label].pop())

    train_file.close()
    validation_file.close()
    return 0


def main(argv):
    """ main """
    len_argv = len(argv)
    if len_argv not in (2, 3):
        print(
            "usage: pascal_cropper.py /path/of/VOC<year> /path/of/output train=True",
            file=sys.stderr)
        return 1

    train = len_argv == 3

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

    if train and os.path.exists(out_path + "/ts.csv") and not os.path.exists(
            out_path + "/train.csv"):
        return split_dataset(out_path)

    with open(out_path + '/ts.csv', mode='w') as csv_file:
        # header
        writer = csv.DictWriter(csv_file, field_names)
        writer.writeheader()

        images_path = argv[0] + "/JPEGImages/"

        with cd(argv[0] + "/Annotations/"):
            i = 0
            for image_xml in os.listdir("."):
                tree = etree.parse(image_xml)
                root = tree.getroot()
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                for obj in root.iter('object'):
                    # skip difficult & object.name not in classes
                    label = obj.find('name').text

                    if train:
                        if label not in classes or int(obj.find(
                            'difficult').text) == 1:
                            continue
                    else:
                        # some image in the test dataset doesn't have
                        # the difficult parameter. Check only the class
                        if label not in classes:
                            continue

                    bb = obj.find('bndbox')
                    rect = [0, 0, 0, 0]
                    #y1
                    rect[0] = int(float(bb.find('ymin').text))
                    #y2
                    rect[1] = int(float(bb.find('ymax').text))
                    #x1
                    rect[2] = int(float(bb.find('xmin').text))
                    #x2
                    rect[3] = int(float(bb.find('xmax').text))

                    image_file = image_xml.replace(".xml", ".jpg")
                    roi = crop(images_path + image_file, rect)

                    width, height = roi.shape[1], roi.shape[0]

                    avg_shape[0] += width
                    avg_shape[1] += height

                    i += 1

                    # use numeric id for label in csv
                    label_id = classes.index(label)

                    # save file and append row to csv
                    cv2.imwrite(out_path + "/" + image_file, roi)
                    writer.writerow({"file": image_file,
                                     "width": width,
                                     "height": height,
                                     "label": label_id})

            print("Average width & height")
            print(int(float(avg_shape[0] / i)), int(float(avg_shape[1] / i)))

            return split_dataset(out_path) if train else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
