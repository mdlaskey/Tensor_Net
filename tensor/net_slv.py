"""
    Model for net3
        conv
        relu
        fc
        relu
        fc
        tanh
"""


import tensorflow as tf
import inputdata
import random
from tensornet import TensorNet
from alan.lfd_slware.options import SLVOptions
import time
import datetime

class NetSLV(TensorNet):

    def get_x_err(self,y_,y_out):
        return tf.reduce_mean(tf.sqrt(tf.square((self.y_out[:,1] - self.y_[:,1]))))*SLVOptions.X_MID_RANGE

    def get_y_err(self,y_,y_out):
        return tf.reduce_mean(tf.sqrt(tf.square((self.y_out[:,0] - self.y_[:,0]))))*SLVOptions.Y_MID_RANGE

    def __init__(self):
        self.dir = "./net6/"
        self.name = "slv_net"
        self.channels = 1

        self.x = tf.placeholder('float', shape=[None,220, 220, self.channels])
        self.y_ = tf.placeholder("float", shape=[None, 2])


        self.w_conv1 = self.weight_variable([7, 7, self.channels, 5])
        self.b_conv1 = self.bias_variable([5])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x, self.w_conv1) + self.b_conv1)

        conv_num_nodes = self.reduce_shape(self.h_conv1.get_shape())
        fc1_num_nodes = 60
        
        self.w_fc1 = self.weight_variable([conv_num_nodes, fc1_num_nodes])
        # self.w_fc1 = self.weight_variable([1000, fc1_num_nodes])
        self.b_fc1 = self.bias_variable([fc1_num_nodes])

        self.h_conv_flat = tf.reshape(self.h_conv1, [-1, conv_num_nodes])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv_flat, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([fc1_num_nodes, 2])
        self.b_fc2 = self.bias_variable([2])

        self.y_out = tf.tanh(tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(.5*tf.sqrt(tf.square(self.y_out - self.y_)))
        self.g_x = self.get_x_err(self.y_,self.y_out)
        self.g_y = self.get_y_err(self.y_,self.y_out)


        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)


