http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html

https://github.com/asrivat1/deep_q_rl


Description of the algorithm:

Choose action at each time step from Q based on epsilon greedy strategy

Store experiences e_t = (s_t, a_t, r_t, s_t+1) in dataset D_t = {e_1,...,e_t} with max capacity N

Apply minibatch updates on samples drawn from D

Could also clip r + y max Q(s', a', theta-) - Q(s, a, theta) to be between -1 and 1


Algorithm:

Initialize replay memory D to capacity N

Initialize Q with random weights theta

Initialize target Q_hat with weights theta- = theta

For episode = 1:M
    
    Initialize sequence s_1 = {x_1} and preprocessed sequence phi_1 = phi(s_1)

    For t = 1:T

        With probability epsilon select random action a_t

        Else select a_t = argmax_a Q(phi(s_t), a; theta)

        Execute a_t and observe reward r_t and image x_t+1

        Set s_t+1 = s_t, a_t, x_t+1 and preprocess phi_t+1 = phi(s_t+1)

        Store transition (phi_t, a_t, r_t, phi_t+1) in D

        Sample a minibatch of transitions e_j from D

        If episode terminates at step j + 1
            Set y_j = r_j
        else
            set y_j = r_j + gamma * max_a' Q_hat(phi_j+1, a'; theta-)

        Perform gradient descent step on (y_j - Q(phi_j, a_j; theta))^2 wrt to network parameters theta

        Every C steps reset Q_hat = Q
