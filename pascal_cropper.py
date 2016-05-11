"""./pascal_cropper.py PASCAL_2012/VOCdevkit/VOC2012 cropped_dataset
Extracts the 20 categories from the PASCAL dataset (argv[1]).
Crop every image to annotated bounding boxes as described in argv[1]/Annotations/
Creates the cropped_dataset/ts.csv file.
Outputs the average input widht and height
"""
import xml.etree.ElementTree as etree
import sys
import os
import cv2
import csv

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


def main(argv):
    """ main """
    if len(argv) != 2:
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

    with open(out_path + '/ts.csv', mode='w') as csv_file:
        # header
        field_names = ["file", "width", "height", "label"]
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
                    if int(obj.find(
                            'difficult').text) == 1 or label not in classes:
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

                    # save file and append line to csv
                    cv2.imwrite(out_path + "/" + image_file, roi)
                    writer.writerow({"file": image_file,
                                     "width": width,
                                     "height": height,
                                     "label": label})

            print("Average width & height")
            print(int(float(avg_shape[0] / i)), int(float(avg_shape[1] / i)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
