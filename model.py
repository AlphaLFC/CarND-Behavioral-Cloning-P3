#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:56:32 2017

@author: alpha
"""

import os
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Merge
from keras.layers import Dropout, Flatten, Lambda, AveragePooling2D
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
import tensorflow as tf

assert K.image_dim_ordering() == 'tf', 'Image array should be in tf mode.'

def imgpath_to_arr(path):
    '''Read an image array from a path'''
    path = path.strip()
    if path.endswith('_'):
        path = path[:-1]
        img = np.fliplr(mpimg.imread(path))
    else:
        img = mpimg.imread(path)
    return img

def data_generator(df, batch_size=64, only_center=False,
                   use_throttle=False, flip_lr=True):
    '''Data generator with flipping augmented, input should be a dataframe'''
    center_paths = df[0].values
    left_paths = df[1].values
    right_paths = df[2].values
    steer_angles = df[3].values
    throttles = df[4].values
    if only_center:
        all_paths = center_paths
        all_angles = steer_angles
        all_throttles = throttles
    else:
        all_paths = np.hstack([center_paths, left_paths, right_paths])
        all_angles = np.hstack([steer_angles]*3)
        all_throttles = np.hstack([throttles]*3)
    if flip_lr:
        flipped_paths = np.array([path+'_' for path in all_paths])
        flipped_angles = -1 * all_angles
        all_paths = np.hstack([all_paths, flipped_paths])
        all_angles = np.hstack([all_angles, flipped_angles])
        all_throttles = np.hstack([all_throttles]*2)
    if use_throttle:
        all_paths, all_angles, all_throttles = shuffle(all_paths, all_angles,
                                                       all_throttles)
    else:
        all_paths, all_angles = shuffle(all_paths, all_angles)
    # yield batches
    start = 0
    n_samples = len(all_paths)
    while True:
        end = start + batch_size
        batch_paths = all_paths[start:end]
        batch_x = np.array([imgpath_to_arr(path) for path in batch_paths])
        batch_y = all_angles[start:end]
        if use_throttle:
            batch_t = all_throttles[start:end]
            batch_y = np.vstack([batch_y, batch_t]).T
        start += batch_size
        if start >= n_samples:
            start = 0
            if use_throttle:
                all_paths, all_angles, all_throttles = shuffle(all_paths,
                                                               all_angles,
                                                               all_throttles)
            else:
                all_paths, all_angles = shuffle(all_paths, all_angles)
        batch_x, batch_y = shuffle(batch_x, batch_y)
        yield batch_x, batch_y

def normalize(x):
    '''x should be a tensor'''
    x = K.cast(x, dtype='float32')
    normed = -0.5 + x / 255.0
    return normed

def res_block(inlayer, name, regularizer=None):
    '''Inlayer should be a keras layer'''
    ch = inlayer.get_shape().as_list()[-1]
    conv1 = Conv2D(ch, 3, 3, border_mode='same', W_regularizer=regularizer,
                   activation='relu', name=name+'_conv1')(inlayer)
    conv1 = BatchNormalization(name=name+'_conv1_bn')(conv1)
    conv2 = Conv2D(ch, 3, 3, border_mode='same', W_regularizer=regularizer,
                   activation='relu', name=name+'_conv2')(conv1)
    conv2 = BatchNormalization(name=name+'_conv2_bn')(conv2)
    # res = merge([inlayer, conv2], mode='sum', name=name+'_sum')
    res = Merge(mode='sum', name=name+'_sum')([inlayer, conv2])
    return res

def resnet(features, regularizer=None):
    '''Features should be a keras Input layer'''
    normed = Lambda(normalize, name='normed')(features)
    resized = AveragePooling2D((4, 4), name='resized')(normed)
    conv1 = Conv2D(16, 5, 5, border_mode='same', W_regularizer=regularizer,
                   activation='relu', name='conv1')(resized)
    conv1 = BatchNormalization(name='conv1_bn')(conv1)
    pool1 = MaxPooling2D((2, 2), name='pool1')(conv1)
    conv2 = Conv2D(32, 3, 3, border_mode='same', W_regularizer=regularizer,
                   activation='relu', name='conv2')(pool1)
    conv2 = BatchNormalization(name='conv2_bn')(conv2)
    res1 = res_block(inlayer=conv2, regularizer=regularizer, name='res1')
    pool2 = MaxPooling2D((2, 2), name='pool2')(res1)
    conv3 = Conv2D(64, 3, 3, border_mode='same', W_regularizer=regularizer,
                   activation='relu', name='conv3')(pool2)
    conv3 = BatchNormalization(name='conv3_bn')(conv3)
    res2 = res_block(inlayer=conv3, regularizer=regularizer, name='res2')
    pool3 = MaxPooling2D((2, 2), name='pool3')(res2)
    conv4 = Conv2D(128, 3, 3, border_mode='same', W_regularizer=regularizer,
                   activation='relu', name='conv4')(pool3)
    conv4 = BatchNormalization(name='conv4_bn')(conv4)
    res3 = res_block(inlayer=conv4, regularizer=regularizer, name='res3')
    conv5 = Conv2D(128, 3, 3, W_regularizer=regularizer,
                   activation='relu', name='conv5')(res3)
    conv5 = BatchNormalization(name='conv5_bn')(conv5)
    conv6 = Conv2D(128, 3, 3, W_regularizer=regularizer,
                   activation='relu', name='conv6')(conv5)
    conv6 = BatchNormalization(name='conv6_bn')(conv6)
    flatten = Flatten(name='flatten')(conv6)
    return flatten

def regnet(inlayer, regularizer=None, out_dim=1):
    '''Regression model, inlayer should be flattened layer'''
    fc1 = Dense(128, activation='relu', name='fc1',
                W_regularizer=regularizer)(inlayer)
    fc1 = Dropout(0.5, name='fc1_dropout')(fc1)
    fc2 = Dense(64, activation='relu', name='fc2',
                W_regularizer=regularizer)(fc1)
    fc2 = Dropout(0.5, name='fc2_dropout')(fc2)
    fc3 = Dense(16, activation='relu', name='fc3',
                W_regularizer=regularizer)(fc2)
    fc3 = Dropout(0.5, name='fc3_dropout')(fc3)
    predicts = Dense(out_dim, name='predicts',
                     W_regularizer=regularizer)(fc3)
    return predicts


class RegressionNet(object):
    '''Base template.'''
    def __init__(self):
        self._load_data()
        self._add_model()

    def _load_data(self):
        raise NotImplementedError

    def _add_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class AutoSteeringWheel(RegressionNet):
    '''Design an auto steering wheel.'''
    def __init__(self, config):
        self.config = config
        self.regularizer = l2(self.config.l2_weight_decay)
        self.input_shape = (160, 320, 3)
        super(AutoSteeringWheel, self).__init__()

    def _load_data(self):
        df = pd.read_csv(self.config.train_csv_file, header=None)
        train_df, valid_df = train_test_split(df, test_size=0.2)
        self.train_datagen = data_generator(train_df, self.config.batch_size)
        self.valid_datagen = data_generator(valid_df, self.config.batch_size)
        self.train_n_samples = train_df.shape[0] * 6
        self.valid_n_samples = valid_df.shape[0] * 6
        test_df = pd.read_csv(self.config.test_csv_file, header=None)
        self.test_datagen = data_generator(test_df, self.config.batch_size,
                                           only_center=True, flip_lr=False)
        self.test_n_samples = test_df.shape[0]

    def _add_model(self):
        features = Input(shape=self.input_shape, name='features')
        flatten = resnet(features, self.regularizer)
        predicts = regnet(flatten, self.regularizer, out_dim=1)
        self.model = Model(features, predicts)
        no_train_layers = self.model.layers[:3]
        for layer in no_train_layers:
            layer.trainable = False

    def show_model(self, to_png=True):
        for layer in self.model.layers:
            if hasattr(layer, 'trainable'):
                print('{:13s}\t{}'.format(layer.name, layer.trainable))
        if to_png:
            png_path = self.config.save_model_name+'.png'
            print('Painting model to {}'.format(png_path))
            plot(self.model, show_shapes=True, to_file=png_path)

    def train(self, fine_tune=True, save_hist=True):
        '''Train or fine_tune the model.'''
        weights_path = self.config.save_model_name+'.h5'
        if fine_tune and os.path.exists(weights_path):
            print('Loading {}'.format(weights_path))
            self.model.load_weights(weights_path)
            adam_lr = self.config.fine_tune_lr
        else:
            if fine_tune:
                print('No previous saved weights exist, train from scratch.')
            adam_lr = self.config.start_lr
        print('Using Adam optimizer with lr={:f}'.format(adam_lr))
        optimizer = Adam(lr=adam_lr)
        # compile model
        self.model.compile(optimizer, 'mse')
        # train ops
        hist = self.model.fit_generator(self.train_datagen,
                                        samples_per_epoch=self.train_n_samples,
                                        nb_epoch=self.config.epoches,
                                        validation_data=self.valid_datagen,
                                        nb_val_samples=self.valid_n_samples)
        if save_hist:
            train_loss = hist.history['loss']
            valid_loss = hist.history['val_loss']
            epoch_range = range(1, self.config.epoches+1)
            plt.figure(figsize=(8, 5))
            plt.plot(epoch_range, train_loss, label='train_loss')
            plt.plot(epoch_range, valid_loss, label='valid_loss')
            plt.xlabel('epoches')
            plt.ylabel('loss')
            plt.xlim(1, self.config.epoches)
            plt.legend(fontsize=9)
            plt.savefig('log.png')

        self._save_model()

    def _save_model(self):
        net_config = self.model.to_json()
        model_path = self.config.save_model_name+'.json'
        weights_path = self.config.save_model_name+'.h5'
        print('Saving model architechure to {}.'.format(model_path))
        with open(model_path, 'w') as f:
            f.write(net_config)
        print('Saving model weights to {}.'.format(weights_path))
        self.model.save_weights(weights_path)

    def predict(self, img):
        if type(img) is str:
            img = mpimg.imread(img)
        return self.model.predict(img, batch_size=1)

    def evaluate(self):
        self.test_loss = self.model.evaluate_generator(self.test_datagen,
                                                       self.test_n_samples)
        print('Test loss is {:.5f}.'.format(self.test_loss))


class Config:
    batch_size = 128
    epoches = 30  # 10 when fine tuning
    start_lr = 0.001  # default for Adam, just OK when learning from scrath
    fine_tune_lr = 1e-4
    l2_weight_decay = 1e-5
    train_csv_file = 'train/driving_log.csv'
    test_csv_file = 'test/driving_log.csv'
    save_model_name = 'model'

if __name__ == '__main__':
    config = Config()
    tfconf = tf.ConfigProto()
    tfconf.gpu_options.allow_growth = True
    with tf.Session(config=tfconf) as sess:
        K.set_session(sess)
        mywheel = AutoSteeringWheel(config)
        mywheel.show_model()
        mywheel.train()
        mywheel.evaluate()
        test_imgs = np.array([mpimg.imread('test_imgs/'+str(i+1)+'.jpg') for i in range(4)])
        steer_angles = mywheel.predict(test_imgs)
        plt.figure(figsize=(10, 6))
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.imshow(test_imgs[i])
            plt.xticks([]), plt.yticks([])
            plt.title('predict steering angle: {:.5f}'.format(steer_angles[i][0]))
            plt.savefig('test.png')

