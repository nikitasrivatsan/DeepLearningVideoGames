#!/usr/bin/python

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return ft.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bais_variable([64])
    
    W_fc1 = weight_variable([10 * 10 * 64, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, 3)]
    b_fc2 = bias_variable([3])

    s = tf.placeholder("float", [None, 80, 80, 4])

    h_conv1 = tf.nn.relu(conv2d(s_image, W_conv1, 4) + b_conv1)
    h_pool = max_pool(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 10 * 10 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    readout = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    return s, readout

def trainNetwork(s, readout):
    y = r + gamma * tf.reduce_max(
    cost = tf.square(

def playGame():
    s, readout = createNetwork()
