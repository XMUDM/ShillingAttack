# -*- coding: utf-8 -*-
# @Time       : 2019/8/24 10:43
# @Author     : chensi
# @File       : models.py
# @Software   : PyCharm
# @Desciption : None

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import math


class GAN_Attacker:
    def __init__(self):
        print("GAN Attack model")

    def DIS(self, input, inputDim, h, activation, hiddenLayers, _reuse=False):
        # input->hidden
        y, _, W, b = self.FullyConnectedLayer(input, inputDim, h, activation, "dis", 0, reuse=_reuse)

        # stacked hidden layers
        for layer in range(hiddenLayers - 1):
            y, _, W, b = self.FullyConnectedLayer(y, h, h, activation, "dis", layer + 1, reuse=_reuse)

        # hidden -> output
        y, _, W, b = self.FullyConnectedLayer(y, h, 1, "none", "dis", hiddenLayers + 1, reuse=_reuse)

        return y

    def GEN(self, input, num_item, h, outputDim, activation, decay, name="gen", _reuse=False):
        """
        input   :   sparse filler vectors
        output  :   reconstructed selected vector
        """
        # input+thnh
        # input_tanh = tf.nn.tanh(input)

        # input->hidden

        y, L2norm, W, b = self.FullyConnectedLayer(input, num_item, h // decay, activation, name, 0, reuse=_reuse)

        # stacked hidden layers
        h = h // decay
        layer = 0
        # for layer in range(hiddenLayers - 1):
        while True:
            y, this_L2, W, b = self.FullyConnectedLayer(y, h, h // decay, activation, name, layer + 1, reuse=_reuse)
            L2norm = L2norm + this_L2
            layer += 1
            if h // decay > outputDim:
                h = h // decay
            else:
                break
        # hidden -> output
        y, this_L2, W, b = self.FullyConnectedLayer(y, h // decay, outputDim, "none", name, layer + 1, reuse=_reuse)
        L2norm = L2norm + this_L2
        y = tf.nn.sigmoid(y) * 5
        return y, L2norm

    def FullyConnectedLayer(self, input, inputDim, outputDim, activation, model, layer, reuse=False):
        scale1 = math.sqrt(6 / (inputDim + outputDim))

        wName = model + "_W" + str(layer)
        bName = model + "_B" + str(layer)

        with tf.variable_scope(model) as scope:

            if reuse == True:
                scope.reuse_variables()

            W = tf.get_variable(wName, [inputDim, outputDim],
                                initializer=tf.random_uniform_initializer(-scale1, scale1))
            b = tf.get_variable(bName, [outputDim], initializer=tf.random_uniform_initializer(-0.01, 0.01))

            y = tf.matmul(input, W) + b

            L2norm = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            if activation == "none":
                y = tf.identity(y, name="output")
                return y, L2norm, W, b

            elif activation == "sigmoid":
                return tf.nn.sigmoid(y), L2norm, W, b

            elif activation == "tanh":
                return tf.nn.tanh(y), L2norm, W, b
            elif activation == "relu":
                return tf.nn.relu(y), L2norm, W, b
