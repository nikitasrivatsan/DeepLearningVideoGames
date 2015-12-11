#!/usr/bin/python

import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import pong_fun
import random

GAMMA = 0.99 # decay rate of past observations
EPSILON = 1 # initial randomness of actions
ACTIONS = 3 # number of valid actions

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([10 * 10 * 64, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 10 * 10 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    return s, readout

def trainNetwork(s, readout):
    # define the cost function
    r = tf.placeholder("float")
    a = tf.placeholder("float", [ACTIONS])
    y = r + GAMMA * tf.reduce_max(readout)

    cost = tf.square(y - tf.matmul(tf.transpose(readout), a))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # open up a game state
    game_state = pong_fun.GameState()

    # store the previous observations
    D = []

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t, r_0 = game_state.frame_step([1, 0, 0])
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.concatenate(x_t, x_t, x_t, x_t, axis = 2)

    # main loop
    while "pigs" != "fly":
        # choose an action
        a_selected = np.zeros([ACTIONS])
        if random.random() <= EPSILON:
            a_selected[random.random(ACTIONS)] = 1

        # scale down epsilon
        if EPSILON > 0.05:
            EPSILON -= 0.95 / 1000

        a_selected = tf.argmax(readout.eval(feed_dict = {
            s : s_t}))

        # run the selected action and observe next state and reward
        x_t1, r_t = game_state.frame_step(a_selected)
        s_t1 = np.concatenate(x_t1, s_t[:,:,1:], axis = 2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1))
        if len(D) > 1000:
            D.pop(0)

        # sample a minibatch to train on
        minibatch = random.sample(D, 100) if len(D) > 100 else D

        # get the batch variables
        s_j_batch = [d[0] for d in D]
        a_batch = [d[1] for d in D]
        r_batch = [d[2] for d in D]
        s_j1_batch = [d[3] for d in D]

        y_batch = y.eval(feed_dict = {
            a : a_batch,
            r : r_batch,
            s : s_j1_batch})

        # perform gradient step
        train_step.run(feed_dict = {
            y : y_batch,
            s : s_j_batch})

        # update the old values
        s_t = s_t1

def playGame():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout)

def main():
    playGame()

if __name__ == "__main__":
    main()
