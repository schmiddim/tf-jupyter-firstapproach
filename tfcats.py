import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict




def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters(dim):
    tf.set_random_seed(1)
 #   w = tf.get_variable("w", dim, initializer=tf.contrib.layers.xavier_initializer(seed=1))
    w = tf.get_variable("w", [3,12288], initializer=tf.zeros_initializer())
    b = tf.get_variable("b", [3,1], initializer=tf.zeros_initializer())

   # w = np.zeros( (dim,1))
    parameters = {"w": w,
                  "b": b,
                  }
    return parameters




def forward_propagation(X, parameters):
    w = parameters['w']
    print(w)
    b = parameters['b']
    Z3 =tf.matmul(w, X)
    return Z3


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.sigmoid(x=Z3) )
    return cost




