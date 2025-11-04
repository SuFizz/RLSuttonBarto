import numpy as np
import sys
import matplotlib
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class GradientBandits:
    def __init__(self, num_arms, qstar, initial_vals, alpha=0.5, initial_baseline=0):
        self.num_arms = num_arms
        self.qstars = qstar     #np.random.uniform(-2,2,num_arms)
        # print("qstar",self.qstars)
        self.reward_gen = lambda i: np.random.normal(self.qstars[i], 1)
        self.H = np.zeros(num_arms)
        self.avg_rwd = initial_baseline
        self.alpha = alpha

    def __call__(self, qstars=None, step_num=0):
        if qstars is not None and qstars.any():
            self.qstars = qstars
        self.optimal_action = np.argmax(self.qstars) 
        pi = np.exp(self.H - np.max(self.H))
        sigma = np.sum(pi)
        pi /= sigma
        arm_used = np.random.choice(self.num_arms, p=pi) #np.argmax(pi)
        # print(self.H)
        # print(arm_used, self.optimal_action)
        reward_recv = self.reward_gen(arm_used)
        # print(reward_recv)
        self.avg_rwd = ((self.avg_rwd*step_num + reward_recv) * 1.0)/(step_num + 1)
        # print(self.avg_rwd, reward_recv)
        # print(pi)
        # print(pi)
        for a in range(self.num_arms):
            if a==arm_used:
                self.H[a] += self.alpha * (reward_recv - self.avg_rwd) * (1 - pi[a])
                # print ("is arm_used", a, self.H[a])
            else:
                self.H[a] += -self.alpha * (reward_recv - self.avg_rwd) * pi[a]
                # print ("not arm_used", a, self.H[a])
        return arm_used==self.optimal_action, reward_recv
    
if __name__ == "__main__":
    num_repeats = 200
    num_steps = 1000
    num_arms = int(sys.argv[1])
    non_stationary = int(sys.argv[2])
    opt_initial_vals = 5
    avg_rwd_base4_alpha0_1 = np.zeros(num_steps, dtype=float)
    avg_rwd_base0_alpha0_1 = np.zeros(num_steps, dtype=float)
    avg_rwd_base4_alpha0_4 = np.zeros(num_steps, dtype=float)
    avg_rwd_base0_alpha0_4 = np.zeros(num_steps, dtype=float)

    actual_opt_recvd_base4_alpha0_1 = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_base0_alpha0_1 = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_base4_alpha0_4 = np.zeros(num_steps, dtype=float)
    actual_opt_recvd_base0_alpha0_4 = np.zeros(num_steps, dtype=float)
    for j in range(num_repeats):
        dec_epsilon = 1
        qstars = np.random.normal(4, 1, num_arms)
        # print(qstars)

        if non_stationary:
            qstars = np.array([np.random.random()]*10)
        
        B_base4_alpha0_1 = GradientBandits(num_arms, qstars, 4, 0.1, 4)
        B_base0_alpha0_1 = GradientBandits(num_arms, qstars, 0, 0.1, 0)
        B_base4_alpha0_4 = GradientBandits(num_arms, qstars, 4, 0.4, 4)
        B_base0_alpha0_4 = GradientBandits(num_arms, qstars, 0, 0.4, 0)
        
        for i in range(num_steps):
            if non_stationary:
                qstars += np.random.normal(0, 0.01, 10)
            
            opt_rwd_base4_alpha0_1 = B_base4_alpha0_1(qstars=qstars, step_num=i)
            avg_rwd_base4_alpha0_1[i] = (avg_rwd_base4_alpha0_1[i]*j + opt_rwd_base4_alpha0_1[1])/(j+1)
            actual_opt_recvd_base4_alpha0_1[i] += 1 if opt_rwd_base4_alpha0_1[0] else 0

            opt_rwd_base4_alpha0_4 = B_base4_alpha0_4(qstars=qstars, step_num=i)
            avg_rwd_base4_alpha0_4[i] = (avg_rwd_base4_alpha0_4[i]*j + opt_rwd_base4_alpha0_4[1])/(j+1)
            actual_opt_recvd_base4_alpha0_4[i] += 1 if opt_rwd_base4_alpha0_4[0] else 0
            
            opt_rwd_base0_alpha0_1 = B_base0_alpha0_1(qstars=qstars, step_num=i)
            avg_rwd_base0_alpha0_1[i] = (avg_rwd_base0_alpha0_1[i]*j + opt_rwd_base0_alpha0_1[1])/(j+1)
            actual_opt_recvd_base0_alpha0_1[i] += 1 if opt_rwd_base0_alpha0_1[0] else 0

            opt_rwd_base0_alpha0_4 = B_base0_alpha0_4(qstars=qstars, step_num=i)
            avg_rwd_base0_alpha0_4[i] = (avg_rwd_base0_alpha0_4[i]*j + opt_rwd_base0_alpha0_4[1])/(j+1)
            actual_opt_recvd_base0_alpha0_4[i] += 1 if opt_rwd_base0_alpha0_4[0] else 0

    actual_opt_recvd_base0_alpha0_1 /= num_repeats
    actual_opt_recvd_base4_alpha0_1 /= num_repeats
    actual_opt_recvd_base0_alpha0_4 /= num_repeats
    actual_opt_recvd_base4_alpha0_4 /= num_repeats
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # figsize adjusts the figure size

    # Plot on the first subplot (ax1)
    ax1.plot(avg_rwd_base0_alpha0_1, color='k', label = 'b0 alpha0_1')
    ax1.plot(avg_rwd_base0_alpha0_4, color='r', label = 'b0 alpha0_4')
    ax1.plot(avg_rwd_base4_alpha0_1, color='b', label = 'b4 alpha0_1')
    ax1.plot(avg_rwd_base4_alpha0_4, color='g', label = 'b4 alpha0_4')
    ax1.set_title('reward')
    ax1.set_xlabel('Steps')
    ax1.legend()
    ax1.grid(True)

    # Plot on the second subplot (ax2)
    ax2.plot(actual_opt_recvd_base0_alpha0_1, color='k', label = 'b0 alpha0_1')
    ax2.plot(actual_opt_recvd_base0_alpha0_4, color='r', label = 'b0 alpha0_4')
    ax2.plot(actual_opt_recvd_base4_alpha0_1, color='b', label = 'b4 alpha0_1')
    ax2.plot(actual_opt_recvd_base4_alpha0_4, color='g', label = 'b4 alpha0_4')
    ax2.set_title('Opt%')
    ax2.set_xlabel('Steps')
    #ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()