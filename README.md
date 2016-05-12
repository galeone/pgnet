#pgnet

## Dataset 

pgnet uses the PASCAL VOC 2012 train set.

`pascal_cropper.py` generates a new dataset of cropped detected images and a csv file `ts.csv`.

The cropper can handle every dataset that follows the PASCAL VOC 2012 structure.

The script outputs the average size of the cropped images. For the PASCAL VOC 2012 dataset the average size is: 168x184

## Network structure

## Atrous Convolutions
The usage of Atrous Convolutions (Convolution with holes) gives us the possibility to have a small number of parameters with a greater perception field.

This two features in conjuction with the absence of padding, can be used to remove the (max-)pooling step at the end of the convolutional layer.
