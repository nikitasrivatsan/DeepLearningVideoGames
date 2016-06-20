# Using Deep Q Networks to Learn Video Game Strategies
#### Akshay Srivatsan, Ivan Kuznetsov, Willis Wang

## 1. Abstract

In this project, we apply a deep learning model recently developed by Minh et al 2015 [1] to learn optimal control patterns from visual input using reinforcement learning. While this method is highly generalizable, we applied it to the problem of video game strategy, specifically for Pong and Tetris. Given raw pixel values from the screen, we used a convolutional neural network trained with Q learning to approximate future expected reward for any possible action, and then selected an action based on the best possible outcome. We find that this method is capable of generalizing to new problems with no adjustment of the model architecture. After sufficient training, our Pong model achieved better than human performance on the games, demonstrating the potential for deep learning as a powerful and generalizable method for learning high-level control schemes.

## 2. Background

Reinforcement learning develops control patterns by providing feedback on a model’s selected actions, which encourages the model to select better actions in the future. At each time step, given some state s, the model will select an action a, and then observe the new state s' and a reward r based on some optimality criterion.

We specifically used a method known as Q learning, which approximates the maximum expected return for performing an action at a given state using an action-value (Q) function. Specifically, return gives the sum of the rewards until the game terminates, where the reward is discounted by a factor of γ at each time step. We formally define this as:

![alt-text](http://imgur.com/h7MJxSJ.png "(1)")

We then define the action-value function:

![alt-text](http://imgur.com/05MxGxk.png "(2)")

Note that if the optimal Q function is known for state s', we can write the optimal Q function at preceding state s as the maximum expected value of ![alt-text](http://imgur.com/1RSOCHo.png "Sorry, no alt-text for this one"). This identity is known as the Bellman equation:

![alt-text](http://imgur.com/BERyjr2.png "(3)")

The intuition behind reinforcement learning is to continually update the action-value function based on observations using the Bellman equation. It has been shown by Sutton et al 1998 [2] that such update algorithms will converge on the optimal action-value function as time approaches infinity. Based on this, we can define Q as the output of a neural network, which has weights θ, and train this network by minimizing the following loss function at each iteration i:

![alt-text](http://imgur.com/3gFka35.png "(4)")

Where y_i represents the target function we want to approach during each iteration. It is defined as:

![alt-text](http://imgur.com/gKcXJfi.png "(5)")

Note that when i is equal to the final iteration of an episode (colloquially the end of a game), the Q function should be 0 since it is impossible to attain additional reward after the game has ended. Therefore, when i equals the terminal frame of an episode, we can simply write:

![alt-text](http://imgur.com/nU8qRJM.png "(6)")

## 3. Related Work

Traditional approaches use an array to approximate the action-value function. However, this method does not scale well as the complexity of the system increases. Tesauro 1995 [3] proposed TD-Gammon, a method of using a multilayer perceptron with one hidden layer. This method worked well on the backgammon game, but failed to generalize to new problems.

More recently Riedmiller 2005 [4] used a system called neural-fitted Q learning, which used a multilayer perceptron to approximate the Q function. However, the approach had issues as it trained in batch updates with cost proportional to the size of the dataset, which limits the scalability of the system. Also, the tasks which Riedmiller considered had a very low dimensional state space. For the problem of video games, however, it is necessary to consider incredibly high dimensional visual input, and not allow our system to know anything about the internal game state or the rules of the game.

Minh et al 2015 [1] proposed deep Q learning, where a deep convolutional neural network is used to calculate the Q function. Convolutional neural networks are particularly well suited to the visual state space of video games. Furthermore, they also proposed a new method for training the network. While online approaches have a tendency to diverge or overfit, and training on the entire dataset is quite costly, Minh et al 2015 trained on minibatches sampled from a replay memory. The replay memory contains all previously seen state transitions, and the associated action and reward. At each time step a minibatch of transitions is sampled from the replay memory and the loss function described above is minimized for that batch with respect to the network weights.

This approach carries many advantages in addition to the aforementioned computational cost benefits. By using a replay memory, each experience can be used in multiple updates, which means that the data is being used more effectively. Also, using an online approach has the disadvantage that all consecutive updates will be over highly correlated states. This can cause the network to get stuck in a poor local optimum, or to diverge dramatically. Using a batched approach over more varied experiences smooths out the updates and avoids this problem.

## 4. Deep Q Learning Algorithm

The pseudo-code for the Deep Q Learning algorithm, as given in [1], can be found below:

```
Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
	Initialize state s_1
	for t = 1, T do
		With probability ϵ select random action a_t
		otherwise select a_t=argmax_a  Q(s_t,a; θ_i)
		Execute action a_t in emulator and observe r_t and s_(t+1)
		Store transition (s_t,a_t,r_t,s_(t+1)) in D
		Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
		Set y_j:=
			r_j for terminal s_(j+1)
			r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
		Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
	end for
end for
```

## 5. Experimental Methods

The network was trained on the raw pixel values observed from the game at each time step. We preprocessed the images by converting to grayscale, resizing them to 80x80, and then stacked together the last four frames to produce an 80x80x4 input array.

The architecture of the network is described in Figure 1 below. The first layer convolves the input image with an 8x8x4x32 kernel at a stride size of 4. The output is then put through a 2x2 max pooling layer. The second layer convolves with a 4x4x32x64 kernel at a stride of 2. We then max pool again. The third layer convolves with a 3x3x64x64 kernel at a stride of 1. We then max pool one more time. The last hidden layer consists of 256 fully connected ReLU nodes.

![alt-text](http://imgur.com/mfatQrY.png "Figure 1")

The output layer, obtained with a simple matrix multiplication, has the same dimensionality as the number of valid actions which can be performed in the game, where the 0th index always corresponds to doing nothing. The values at this output layer represent the Q function given the input state for each valid action. At each time step, the network performs whichever action corresponds to the highest Q value using a ϵ greedy policy.

At startup, we initialize all weight matrices randomly using a normal distribution with a standard deviation of 0.01. Bias variables are all initialized at 0.01. We then initialize the replay memory with a max size of 500,000 observations.

We start training by choosing actions uniformly at random for 50,000 time steps, without updating the network weights. This allows us to populate the replay memory before training begins. After that, we linearly anneal ϵ from 1 to 0.1 over the course of the next 500,000 frames. During this time, at each time step, the network samples minibatches of size 100 from the replay memory to train on, and performs a gradient step on the loss function described above using the Adam optimization algorithm with a learning rate of 0.000001. After annealing finishes, the network continues to train indefinitely, with ϵ fixed at 0.1.

An Amazon Web Services G2 large instance was used to efficiently conduct training on a GPU. We implemented the DQN in Google’s newly released TensorFlow library.

## 6. Results

See these links for videos of the DQN in action:

[DQN playing a long game of Pong](https://www.youtube.com/watch?v=NE_KKM0e38s)

[Visualization of convolutional layers and Q function](https://www.youtube.com/watch?v=W9jGIzkVCsM)

We found that for Pong, good results were achieved after approximately 1.38 million time steps, which corresponds to about 25 hours of game time. Qualitatively, the network played at the level of an experienced human player, usually beating the game with a score of 20-2. Figure 2 shows the maximum Q value for the first ~20,000 training steps. To minimize clutter, only maximum Q values corresponding to the first 100 training steps of each consecutive group of 1,250 training steps are shown. As can be seen, the maximum Q value increases over time. This indicates improvement as it means that the network is expecting to receive a greater reward per game as it trains for longer. In theory, as we continue to train the network, the value of the maximum Q value should plateau as the network reaches an optimal state.

![alt-text](http://imgur.com/rnLVmyd.png "Figure 2")

The final hidden representation of the input image frames in the network was a 256 dimensional vector. Figure 3 shows t-SNE embedding of ~1,200 such representations which were sampled over a period of 2 hours with fixed network weights which had been learned after 25 hours of training. The points in the visualization were also color-coded based on the maximum Q values at the output layer for that frame.

![alt-text](http://imgur.com/hCj7BCG.png "Figure 3")

The network controlled paddle is on the left and highlighted yellow for easy visualization. The paddle controlled by the “opponent,” a hard-coded AI built into the game, is on the right. As can be seen from the image frames, high maximum Q-values correspond to when the ball was near the opponent's paddle, so the network had a high probability of scoring. Similarly, low maximum Q-values correspond to when the ball was near the network's paddle, so the network had an increased chance of losing the points. When the ball is in the center of the court, a neutral maximum Q value occurs.

We are also currently actively working on getting the system to learn Tetris as well. This section will be updated as further progress in this direction is made.

## 7. Conclusions

We utilized deep Q learning to train a neural network to play Pong and partially to play Tetris from just images of the current game screen and no knowledge of the internal game state. Qualitatively, the trained Pong network appeared to perform near human level. We are still working on training the system to play Tetris, and will hopefully report good results in the near future.

We then went on to show that the Pong network has a relatively high level “understanding” of the current state of the game. For example, the Pong network’s function was maximized when the ball was near the opponent's paddle, when it would be reasonable to expect a point. However, it was minimized when the ball was near the network’s paddle, as there was a chance it could lose the point.

Significant improvements on the current network architecture are possible. For example, our implementation of max pooling may have been unnecessary, as it caused the convolution kernels in the deeper layers to have a size comparable to that of the previous layer itself. Hence, max pooling may have discarded useful information. Given more time, it would be nice to perform more tuning over the various network parameters, such as the learning rate, the batch size, the replay memory, and the lengths of the observation and exploration phases. Additionally, we could check the performance of the network if the game parameters are tweaked (e.g. the angle of the ball’s bounce is changed slightly). This would yield important information about the network’s ability to extrapolate to new situations.

Based on the difference in results between Tetris and Pong, we should also expect the system to converge faster for games which provide rewards (either positive or negative) very frequently, even while the network is acting randomly. This leads us to believe that while our system should theoretically be able to learn any game, it might achieve human-level performance faster for genres like fighting games or other similar types in which the game score changes rapidly over a short amount of time. This might be another interesting future direction to explore.

Ultimately, the results obtained here demonstrate the usefulness of convolutional neural networks to perform deep Q learning as proposed in [1]. However, we have only applied these methods to relatively simple games; CNNs have also been used in much more complex image processing and object recognition. This would indicate that we can apply similar techniques to those described here to much more challenging tasks. Further research well undoubtedly lead to interesting results in this field.

## 8. References

[1] Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level Control through Deep Reinforcement Learning. Nature, 529-33, 2015.

[2] Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, 1998.

[3] Gerald Tesauro. Temporal difference learning and td-gammon. Communications of the ACM, 38(3):58–68, 1995.

[4] Martin Riedmiller. Neural fitted q iteration–first experiences with a data efficient neural reinforcement learning method. In Machine Learning: ECML 2005, pages 317–328. Springer, 2005.
