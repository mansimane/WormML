# Head and Tail Localization of C. elegans

## Steps:

1. Create datasets/wormml folder. This folder should have images in a folder as well as labels in xls file.
2. Create CreateDatasetV2.py - This files reads images and labels, splits images into train and validation data,
detects worm bounding boxes and dumps 150x150 worm images.
3. Run dsnt/wormml_dsnt.py - This file trains the model and dumps the predictions as well as tensorboard summaries in experiments folder.
4. search_hyper_parameters.py - Performs 10-fold validation on the data. Can also be used to search hyper-parameters.



## References:
* Numerical Coordinate Regression with Convolutional Neural Networks: https://arxiv.org/pdf/1801.07372.pdf
* DSNT tensorflow Code(Unofficial): https://github.com/ashwhall/dsnt/blob/master/dsnt.py
* Tensorflow code is inspired from: https://github.com/cs230-stanford