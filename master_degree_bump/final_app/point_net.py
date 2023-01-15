import keras
from tensorflow.keras import layers
import mediapipe as mp
from transformers import FaceMeshCollecter, FaceMeshDrawer, Transformer
from point_net_api import *

def get_base_point_net_model(first_degree: int=128, second_degree: int=256,
                             third_degree: int=512, fourth_degree: int=1024):
    """get base point net model that picks 3D face scan and calculates probability of the
    face scan to belong to one of the 8 emotional classes
    :param first_degree, second_degree, third_degree, fourth_degree: degrees of the model
        layers defining how many neurons will be there per layer
    :return: keras model realizing point net architecture for 8 emotions classification
    """
    first_degree, second_degree, third_degree, fourth_degree = 128, 256, 512, 1024

    inputs = keras.Input(shape=(468, 3))

    #   first stage, make T-net transformation and make two convolution transformations
    x = tnet(inputs, 3, first_degree, second_degree, third_degree, fourth_degree, 'leaky_relu')
    x = conv_bn(x, 64, activation_function='leaky_relu', kernel_size=3)
    x = conv_bn(x, 64, activation_function='leaky_relu', kernel_size=3)
    
    #   second stage, make second T-net transformation and then three convolution transformations
    x = tnet(x, 64, first_degree, second_degree, third_degree, fourth_degree, activation_function='leaky_relu')
    x = conv_bn(x, 64, activation_function='leaky_relu', kernel_size=3)
    x = conv_bn(x, first_degree, activation_function='leaky_relu', kernel_size=3)
    x = conv_bn(x, third_degree, activation_function='leaky_relu', kernel_size=3)
    
    #   third stage, make max pooling amd then perform several dense layers with dropouts to make sure 
    # that there will not be overfitting of the model and stay abstract enough from the training data
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, second_degree, activation_function='leaky_relu')
    x = layers.Dropout(0.25)(x)
    x = dense_bn(x, first_degree, activation_function='leaky_relu')
    x = layers.Dropout(0.25)(x)

    #   final layer that calculates probability of the current record to belong to any of the 8 emotion classes
    outputs = layers.Dense(8, activation="softmax")(x)

    #   form an NN model and give it for use
    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet_class")
    return model

def point_net_weights_load(model: keras.Model, path: str):
    """load pre-trained weights of the model
    :param model: keras model with exact the same structure, but not trained
    :param path: path to the directory where model weights are located
    :return: model with loaded weights
    """
    model.load_weights(path)
    return model