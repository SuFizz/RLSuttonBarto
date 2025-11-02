import numpy as np
import sys
import matplotlib
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Bandits:
    def __init__(self, num_arms, qstar, epsilon):
        self.num_arms = num_arms
        self.qstars = qstar     #np.random.uniform(-2,2,num_arms)
        #print(self.qstars)
        self.epsilon = epsilon
        self.optimal_action = np.argmax(qstar)
        self.reward_gen = lambda i: np.random.normal(self.qstars[i], 1)
        self.q = np.zeros((num_arms,))
        self.num_times_arm_used = np.zeros((num_arms,))

    def __call__(self, epsilon=0, qstars=None):
        self.epsilon = epsilon if epsilon > 0 else self.epsilon
        if qstars is not None and qstars.any():
            self.qstars = qstars
        arm_used = np.argmax(self.q) if np.random.random() > self.epsilon else np.random.randint(0,self.num_arms)
        self.num_times_arm_used[arm_used] += 1
        reward_recv = self.reward_gen(arm_used)
        self.q[arm_used] += (reward_recv - self.q[arm_used])/self.num_times_arm_used[arm_used]
        return arm_used==self.optimal_action, reward_recv
    
if __name__ == "__main__":
    num_repeats = 200
    num_steps = 1000
    num_arms = int(sys.argv[1])
    non_stationary = int(sys.argv[2])
    avg_rwd_greedy = np.zeros(num_steps, dtype=float)
    avg_rwd_eps0_1 = np.zeros(num_steps, dtype=float)
    avg_rwd_eps0_01 = np.zeros(num_steps, dtype=float)
    avg_rwd_eps_decay = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_greedy = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_eps0_1 = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_eps0_01 = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_eps_decay = np.zeros(num_steps, dtype=float)
    max_rwd_recvd = -math.inf
    for j in range(num_repeats):
        dec_epsilon = 1
        qstars = np.random.uniform(-2, 2, int(sys.argv[1]))

        if non_stationary:
            qstars = np.array([np.random.random()]*10)
        
        B_greedy = Bandits(num_arms, qstars, 0)
        B_eps0_1 = Bandits(num_arms, qstars, 0.1)
        B_eps0_01 = Bandits(num_arms, qstars, 0.01)
        B_eps_decay = Bandits(num_arms, qstars, dec_epsilon)
        
        for i in range(num_steps):
            if non_stationary:
                qstars += np.random.normal(0, 0.01, 10)
            
            opt_rwd_greedy = B_greedy(qstars=qstars)
            avg_rwd_greedy[i] = (avg_rwd_greedy[i]*j + opt_rwd_greedy[1])/(j+1)
            actual_opt_recvd_greedy[i] += 1 if opt_rwd_greedy[0] else 0
        
            opt_rwd_eps0_1 = B_eps0_1(qstars=qstars)
            avg_rwd_eps0_1[i] = (avg_rwd_eps0_1[i]*j + opt_rwd_eps0_1[1])/(j+1)
            actual_opt_recvd_eps0_1[i] += 1 if opt_rwd_eps0_1[0] else 0

            opt_rwd_eps0_01 = B_eps0_01(qstars=qstars)
            avg_rwd_eps0_01[i] = (avg_rwd_eps0_01[i]*j + opt_rwd_eps0_01[1])/(j+1)
            actual_opt_recvd_eps0_01[i] += 1 if opt_rwd_eps0_01[0] else 0

            opt_rwd_eps_decay = B_eps_decay(epsilon=dec_epsilon, qstars=qstars)
            avg_rwd_eps_decay[i] = (avg_rwd_eps_decay[i]*j + opt_rwd_eps_decay[1])/(j+1)
            actual_opt_recvd_eps_decay[i] += 1 if opt_rwd_eps_decay[0] else 0
            dec_epsilon *= 0.995 #if opt_rwd_eps_decay[0] else 1
    
    actual_opt_recvd_greedy /= num_repeats
    actual_opt_recvd_eps0_1 /= num_repeats
    actual_opt_recvd_eps0_01 /= num_repeats
    actual_opt_recvd_eps_decay /= num_repeats
    #print(actual_opt_recvd_eps0_01)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # figsize adjusts the figure size

    # Plot on the first subplot (ax1)
    ax1.plot(avg_rwd_greedy, color='k', label = 'greedy')
    ax1.plot(avg_rwd_eps0_1, color='r', label = 'eps0_1')
    ax1.plot(avg_rwd_eps0_01, color='g', label = 'eps0_01')
    ax1.plot(avg_rwd_eps_decay, color='b', label = 'eps_decay')
    ax1.set_title('reward')
    ax1.set_xlabel('Steps')
    ax1.legend()
    ax1.grid(True)

    # Plot on the second subplot (ax2)
    ax2.plot(actual_opt_recvd_greedy, color='k', label = 'greedy')
    ax2.plot(actual_opt_recvd_eps0_1, color='r', label = 'eps0_1')
    ax2.plot(actual_opt_recvd_eps0_01, color='g', label = 'eps0_01')
    ax2.plot(actual_opt_recvd_eps_decay, color='b', label = 'eps_decay')
    ax2.set_title('Opt%')
    ax2.set_xlabel('Steps')
    #ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()