
from __future__ import division, print_function, absolute_import

import tflearn
import h5py
import numpy
# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Data loading
h5f = h5py.File('path/to/hdf5/file/training_va.h5', 'r')
X = h5f['X']
Y=numpy.column_stack((h5f['valence'],h5f['arousal']))
print(Y.shape)

h5f_val = h5py.File('path/to/hdf5/file/test_va.h5', 'r')
testX = h5f_val['X']
testY = numpy.column_stack((h5f_val['valence'],h5f_val['arousal']))
print(testY.shape)

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True, mean=[ 0.38888048 , 0.43694749,  0.53946589])

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([49, 49], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 49, 49, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 2)
# mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
regression = tflearn.regression(net, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
# Training
model = tflearn.DNN(regression,
  tensorboard_dir='Logs/',
  checkpoint_path='Snapshots/model_resnet_',
  max_checkpoints=10,
  clip_gradients=0.)

model.fit(X, Y, n_epoch=200, validation_set=(testX, testY),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnext_va')

h5f.close()
h5f_val.close()