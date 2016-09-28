#pgnet

## Get the dataset

pgnet uses the PASCAL VOC 2012 train & test set, for train and test respectively.

`inputs/pascal_trainval.py` generates a new dataset of cropped detected images and a csv file `ts.csv`.

The cropper can handle every dataset that follows the PASCAL VOC 2012 structure.

Here's some code to get the datasets and to build the cropped dataset.

```
mkdir -p ~/data/PASCAL_2012/test
cd ~/data/PASCAL_2012
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget http://pjreddie.com/media/files/VOCdevkit_18-May-2011.tar
wget http://pjreddie.com/media/files/VOC2012test.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCdevkit_18-May-2011.tar
cd -
python inputs/pascal_trainval.py ~/data/PASCAL_2012/VOCdevkit/VOC2012 ~/data/PASCAL_2012_cropped/
# It could take a long time if you hdd is slow.
# At the end ~/data/PASCAL_2012_cropped/ contains the cropped images and `ts.cs`
# `train.csv` and `validations.csv` are created in the current folder.
cd ~/data/PASCAL_2012/test
mv ../VOC2012test.tar .
tar xf VOC2012test.tar
cd ~
```

## Train the network

Hith: lunch it into a tmux/screen session

```
python train.py
```

Train will stop when the average validation accuracy stops to increase for a predefined number of epoch (see the constant in the `train.py` file).

If you want to increase the accuracy, restart the train dropping the `keep_prob_` value in the train step.


## Test

Classification task: use realtive frequencies to classify images

```
python test_classification_pascal_rf.py --test-ds ~/data/PASCAL_2012/test/VOCdevkit/VOC2012/
```

Localization task: use relative frequencies to detect regions in the input image.

```
python test_localization_pascal_rf.py --test-ds ~/data/PASCAL_2012/test/VOCdevkit/VOC2012/
```

The `results` folder will contain the PASCAL VOC 2012 competition 1 and 3 results.

## Submit the reults

Creates the archives with `tar -cvzf results.tgz results` and submit the result to the evaluation server: http://host.robots.ox.ac.uk:8080/
