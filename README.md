# HAFR
This our official implementation for the paper: Hierarchical Attention Network for Visually-aware
Food Recommendation

If you use the codes, please kindly cite our paper. Thanks!

## Environment
Python 2.7 <br>
TensorFlow >= 1.4.0

## Quick Start
This command shows the effect of HAFR on pretrained model for dataset in epoch 300. <br>
```
python HAFR.py --pretrain 1 --reg 0.1 --reg_image 0.01 --reg_h 1 --reg_w 1 
```

## Dataset
We provide processed dataset: Allrecipes in [here](https://www.kaggle.com/elisaxxygao/foodrecsysv1) <br>

**data.train.rating** 
* Train file.
* Each line is a training instance: userID\t itemID\t rating\t timestamp

**data.test.rating**
* Test file.
* Each line is a testing instance: userID\t itemID\t rating\t timestamp 

**data.test.negative**
* Test file (negative instances).
* Each line corresponds to the user of test.rating, containing 500 negative samples.
* Each line is in the format: (userID:itemID1,itemID2...)\t negativeItemID1\t negativeItemID2...

**data.valid.rating**
* Valid file.
* Each line is a validation instance: userID\t itemID\t rating\t timestamp 

**data.valid.negative**
* Valid file (negative instances).
* Each line corresponds to the user of valid.rating, containing 500 negative samples.
* Each line is in the format: (userID:itemID1,itemID2...)\t negativeItemID1\t negativeItemID2...
