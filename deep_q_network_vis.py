#!/usr/bin/python

import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import pong_fun as game # whichever is imported "as game" will be used
import dummy_game
import random
import numpy as np
sys.path.append("VisTools/")
import LiveVisTools as vis

GAMMA = 0.99 # decay rate of past observations
ACTIONS = 3 # number of valid actions
OBSERVE = 200. # timesteps to observe before training
EXPLORE = 10. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 500000 # number of previous transitions to remember
BATCH = 100 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
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
    
    W_fc1 = weight_variable([256, 256])
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, ACTIONS])
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

    h_pool3_flat = tf.reshape(h_pool3, [-1, 256])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_conv1, h_pool1,  h_conv2, h_pool2, h_conv3, h_pool3, h_pool3_flat, h_fc1

def trainNetwork(s, readout, h_conv1, h_pool1,  h_conv2, h_pool2, h_conv3, h_pool3, h_pool3_flat, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    #train_step = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(cost)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    vis_state = vis.plot_state()

    # store the previous observations in replay memory
    D = []

    # printing
    #a_file = open("logs/readout.txt", 'w')
    #h_file = open("logs/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    x_t, r_0, terminal = game_state.frame_step([1, 0, 0])
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    epsilon = INITIAL_EPSILON
    t = 0

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path

    
    while "pigs" != "fly":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

##        # DEBUG
##        print readout_t,
##        print a_t

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            
        for i in range(0, K):
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.pop(0)

        #Update graphs
        vis_state.update(r_t,np.max(readout_t),h_conv1.eval(feed_dict={s:[s_t]})[0],h_conv2.eval(feed_dict={s:[s_t]})[0],h_conv3.eval(feed_dict={s:[s_t]})[0])
            

##        # only train if done observing
##        if t > OBSERVE:
##            # sample a minibatch to train on
##            minibatch = random.sample(D, BATCH)
##
##            # get the batch variables
##            s_j_batch = [d[0] for d in minibatch]
##            a_batch = [d[1] for d in minibatch]
##            r_batch = [d[2] for d in minibatch]
##            s_j1_batch = [d[3] for d in minibatch]
##
##            y_batch = []
##            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
##            for i in range(0, len(minibatch)):
##                # if terminal only equals reward
##                if minibatch[i][4]:
##                    y_batch.append(r_batch[i])
##                else:
##                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
##
##            # perform gradient step
##            train_step.run(feed_dict = {
##                y : y_batch,
##                a : a_batch,
##                s : s_j_batch})

        # update the old values
        s_t = s_t1
        t += K

        # save progress every 10000 iterations
        #if t % 10000 == 0:
        #    saver.save(sess, 'saved_networks/pong-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        #print "TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)

        #print h_conv3.eval(feed_dict={s:[s_t]})[0].shape

        # write info to files
        #if t % 1000 == 0:
            #a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #cv2.imwrite("logs/frame" + str(t) + ".png", x_t1_col)

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_conv1, h_pool1,  h_conv2, h_pool2, h_conv3, h_pool3, h_pool3_flat, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_conv1, h_pool1,  h_conv2, h_pool2, h_conv3, h_pool3, h_pool3_flat, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
