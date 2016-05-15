#pgnet

## Dataset 

pgnet uses the PASCAL VOC 2012 train set.

`pascal_cropper.py` generates a new dataset of cropped detected images and a csv file `ts.csv`.

The cropper can handle every dataset that follows the PASCAL VOC 2012 structure.

The script outputs the average size of the cropped images. For the PASCAL VOC 2012 dataset the average size is: 168x184.

Here's some code:

```
mkdir -p ~/data/PASCAL_2012
cd ~/data
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget http://pjreddie.com/media/files/VOCdevkit_18-May-2011.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCdevkit_18-May-2011.tar
cd -
python pascal_cropper.py ~/data/PASCAL_2012/VOCdevkit/VOC2012 ~/data/PASCAL_2012_cropped/
# It could take a long time if you hdd is slow.
# At the end ~/data/PASCAL_2012_cropped/ contains the cropped images and `train.csv` and `validations.csv`
```

## Network structure

## Atrous Convolutions
The usage of Atrous Convolutions (Convolution with holes) gives us the possibility to have a small number of parameters with a greater perception field.

This two features in conjuction with the absence of padding, can be used to remove the (max-)pooling step at the end of the convolutional layer.
