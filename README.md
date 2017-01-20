# Behavioral Cloning

## Data preparation

The car was trained in both tracks, and the training dataset consist of records of about several laps of normal, centerline driving and records of recovering from either left or right road side. Additionally, a testing dataset with only records of normal driving in the first track were created to evaluate the trained model. 

A `data_generator` function was designed to generate batches of image arrays and labels from the \<csv log file\>. In model training process, the original training dataset were split into train/valid parts with a proportion of 0.8:0.2. And a simple data augmentation technique that **left-right flipping both images and steering angles** was adapted. Also, left, right and center camera images were all used in training. Thus, in training process, the `data_generator` will provide batches of left-right flipped and multi-camera image arrays and steering angles. While in evaluating the model on testing dataset, the `data_generator` will only provide center camera images without left-right flipping.

Note that the `data_generator` will not do data preprocessing. Image resizing operation as well as normalization are integrated in the model architechture.

## Network architechture

I designed a resnet-like neural network. The architechture consists of three major parts, preprocessing layers, feature learning layers and regression layers. 
- The preprocessing layers include normalization and resizing operation. An `AveragePooling2D` layer was used as
- The feature learning layers include 3 residual blocks and a total of 12 convolutional layers
## Model training strategy
