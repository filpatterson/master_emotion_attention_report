import os.path
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LeakyReLU


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features: int, l2reg: float = 0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        """overwrite of the call function for the class
        :param x: input
        :return: regularization result
        """
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def read_all_pickles(package_name: str = 'original_pkls', prefix_names: list = None, file_specification: str = None,
                     top_value: int = 20000) -> pd.DataFrame:
    """read all pickle files with specification of the dataframe with facial scans
    :param package_name: name or path of the package where pickle files are located
    :param prefix_names: specify list of some unique files required to load
    :param file_specification: what kind of file it is required to read
    :param top_value: considering that files are presented in a form of package_name/prefix_name+PID+file_specification
                        it is required to set max possible PID value till which function will scan for pickles
    :return: dataframe of all scanned pickles
    """
    global_df = None
    file_postfix = '_landmarks.pkl'
    if file_specification is not None:
        file_postfix = file_specification

    #   go through base files
    for index in range(top_value):
        current_path = package_name + '/' + str(index) + file_postfix
        if os.path.isfile(current_path):
            cur_df = pd.read_pickle(current_path)
            global_df = pd.concat([global_df, cur_df])

    #   go through prefix files if there are any specified
    if prefix_names is not None:
        for prefix in prefix_names:
            for index in range(top_value):
                current_path = package_name = '/' + prefix + str(index) + file_postfix
                if os.path.isfile(current_path):
                    cur_df = pd.read_pickle(current_path)
                    global_df = pd.concat([global_df, cur_df])
    return global_df


def conv_bn(x, filters: int, activation_function: str = "relu", kernel_size: int = 1):
    """1D convolution with batch normalization, setting specified activation function
    :param x: input data
    :param filters: how many filters apply for convolution (how many neurons)
    :param activation_function: what kind of activation function to apply for layer
    :param kernel_size: size of the convolution to consider
    :return: Convolutional layer with batch normalization, specified activation and size of kernel
    """
    if activation_function == 'leaky_relu':
        x = layers.Conv1D(filters, kernel_size=kernel_size, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return LeakyReLU()(x)
    else:
        x = layers.Conv1D(filters, kernel_size=kernel_size, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation(activation_function)(x)


def dense_bn(x, filters: int, activation_function: str = 'relu'):
    """Dense layer with specification of how many neurons and what activation function to use
    :param x: input data
    :param filters: how many filters to use (or how many neurons in layer)
    :param activation_function: activation function for neuron, defaults to 'relu'
    :return: Dense layer with specified neurons count and activation function
    """
    if activation_function == 'leaky_relu':
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return LeakyReLU()(x)
    else:
        return layers.Activation(activation_function)(x)


def tnet(inputs, num_features: int, first_degree: int, second_degree: int, third_degree: int, fourth_degree: int,
         activation_function: str='relu'):
    """T-net layer that performs multiple 1D convolutions with Max Pooling of received results to reduce
    dimensionality and complexity of the problem. Final layers are represented by Dense layers.
    Results that will be received after all those transformations will be dot of original matrix
    with new one.
    :param inputs: input data
    :param num_features: dimensionality of the input data
    :params first_degree, second_degree, third_degree, fourth_degree: how many neurons to set for specific layers,
                                                        try to use numbers with base 2
    :param activation_function: what kind of activation to apply for every layer of the convolution or dense, defaults
                                to the 'relu' value
    :return: dot product of the original matrix with updated one.
    """

    #   Initialize bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32, activation_function=activation_function)
    x = conv_bn(x, first_degree, activation_function=activation_function)
    x = conv_bn(x, fourth_degree, activation_function=activation_function)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, third_degree, activation_function=activation_function)
    x = dense_bn(x, second_degree, activation_function=activation_function)
    x = layers.Dense(
        num_features * num_features, kernel_initializer="zeros", bias_initializer=bias, activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)

    return layers.Dot(axes=(2, 1))([inputs, feat_T])
