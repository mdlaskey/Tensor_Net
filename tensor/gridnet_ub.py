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
import time
import datetime
import IPython
import numpy as np

class GridNet_UB(TensorNet):

    def get_acc_w(self,y_,y_out,weights):
        cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
       
        ac = tf.reduce_mean(tf.cast(cp, tf.float32)*weights)
    
        return ac


    def get_acc(self,y_,y_out):
        cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
       
        ac = tf.reduce_mean(tf.cast(cp, tf.float32))
    
        return ac


    def get_prob(self,y_out):
        v = tf.argmax(y_out,1)
        return y_out[:,v]

    def get_weight(self,trajectory):
        T = len(trajectory)
        weight = 1
        #print "T ",T
        if(T == 1):
            return weight

        for t in range(1,T):

            state = trajectory[t][0]
            state = state+np.zeros([1,2])
            y = trajectory[t][1]
            v = np.argmax(y)
          
            dist = self.dist(state)
            v_ = np.argmax(dist)
            
            if(v_ == v):
                #print t
                weight = 1.0*weight
            else:
                return 0.0
            #weight = weight*dist[v]/1.0

        return weight




    def __init__(self):
        self.dir = "./gridnet/"
        self.name = "gridnet"
        
        self.x = tf.placeholder('float', shape=[None, 2])
        self.y_ = tf.placeholder("float", shape=[None, 5])
        self.weights = tf.placeholder("float", shape=[None,1])


        self.w_fc1 = self.weight_variable([2, 5])
        self.b_fc1 = self.bias_variable([5])

        self.h_1 = tf.nn.relu(tf.matmul(self.x, self.w_fc1) + self.b_fc1)

        self.w_fc2 = self.weight_variable([5, 5])
        self.b_fc2 = self.bias_variable([5])

        self.y_out = tf.nn.softmax(tf.matmul(self.h_1, self.w_fc2) + self.b_fc2)
        
        #self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_out)*self.weights, reduction_indices=[1]))
        
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_out), reduction_indices=[1]))
        self.acc_w = self.get_acc_w(self.y_,self.y_out,self.weights)
        self.acc = self.get_acc(self.y_,self.y_out)
        self.train_step = tf.train.MomentumOptimizer(.003, .9)
        self.train = self.train_step.minimize(self.loss)

