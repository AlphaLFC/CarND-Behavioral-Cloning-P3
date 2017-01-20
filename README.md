# Behavioral Cloning Project Documentation

## Data preparation

The car was trained in both tracks, and the training dataset consist of records of about several laps of normal, centerline driving and records of recovering from either left or right road side. Additionally, a testing dataset with only records of normal driving in the first track were created to evaluate the trained model. 

A `data_generator` function was designed to generate batches of image arrays and labels from the \<csv log file\>. In model training process, the original training dataset were split into train/valid parts with a proportion of 0.8:0.2. And a simple data augmentation technique that **left-right flipping both images and steering angles** was adopted. Also, left, right and center camera images were all used in training. Thus, in training process, the `data_generator` will provide batches of left-right flipped and multi-camera image arrays and steering angles. While in evaluating the model on testing dataset, the `data_generator` will only provide center camera images without left-right flipping.

Note that the `data_generator` will not do data preprocessing. Image resizing operation as well as normalization are integrated in the model architechture.

## Network architechture

I designed a resnet-like neural network. The architechture consists of three major parts, **preprocessing layers, feature learning layers and regression layers**. Descriptions as below.

- The preprocessing layers are designed for image preprocessing, including normalization and resizing operation. A `Lambda` layer was used to insert the normalization function. An `AveragePooling2D` layer was adopted to resize the camera image from shape (160, 320, 3) to (40, 80, 3).
- The feature learning layers are designed for feature extraction purpose, they should extract features like road edges after sufficient learning. These layers include 3 residual blocks (`res_block`) and a total of 12 `Convolution2D` layers. A `res_block` consists of two `Convolutional2D` layer. Each `Convolution2D` layer is followed by a `BatchNormalization` layer, for training acceleration， normalization and regularization. Except `conv1` using a 5x5 kernel, rest convolutional layers all take 3x3 filters. Detailed structure is shown in the figure below.
- The regression layers are going to learn how to predict steering angles. They are simply 4 fully connected `Dense` layers. `Dropout` layers are added to prevent overfitting. This part is much like the structure in the Nvidia paper.

When designing for convolutional and fully connected layers with trainable weights, I added a L2 weight decay to each of them to prevent overfitting. 

I plotted the model architechture with layer shape shown using keras, as the figure below.
![Model architechture](model.png)

## Model training

I adopted the `mse` loss and the `Adam` optimizer for the model. If firstly training from scratch, the starting learning rate (lr) is set to be the default value of 0.001. While for fine tuning a pre-trained model, the starting lr is set to be 0.0001.
Some hyperparameters are as follows.
```python
batch_size = 128 \# 
epoches = 30
start_lr = 0.001
fine_tune_lr = 1e-4
l2_weight_decay = 1e-5
```

## Model evaluatation and simulation tests
