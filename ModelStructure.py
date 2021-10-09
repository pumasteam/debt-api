from tensorflow import keras
from tensorflow.keras.layers import *

class RegressorNN(keras.Model):

    def __init__(self, n_features):
        super(RegressorNN, self).__init__()
        self.dc1 = Dense(n_features*8, activation = 'relu', input_shape = (n_features,))
        self.dc2 = Dense(n_features*4, activation = 'relu')
        self.dc3 = Dense(n_features, activation = 'relu')
        self.outputs = Dense(1)
    def call(self, inputs):
        x = self.dc1(inputs)
        x = self.dc2(x)
        x = self.dc3(x)
        return self.outputs(x)