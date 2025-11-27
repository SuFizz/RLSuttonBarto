import numpy as np

V_TD_error = np.zeros(21)
actions = [-1,1]
start_state = 10
rwd_state = 19
non_state = -1
n = 9
np.random.seed(42)
alpha = 0.1

for episodes in range(1000):
    print (f"\r episodes {episodes}", flush=True, end="")
    t = 0
    T = np.inf
    curr_state = start_state
    states = []
    states.append(curr_state)
    rwds = []
    rwds.append(0)
    while True:
        reward = 0
        if t<T:
            act = np.random.choice(actions)
            if curr_state + act == rwd_state:
                T = t + 1
                reward = 1
            elif curr_state + act == non_state:
                T = t + 1
                reward = 0
            else:
                reward = -1
            curr_state += act
            states.append(curr_state)
            rwds.append(reward)
        tau = t - n + 1
        if tau >= 0:
            #G = (min(tau+n, T) - (tau+1))*(-1)
            G = np.sum(rwds[tau+1: 1+min(tau+n, T)])
            if tau +n < T:
                G += V_TD_error[int(states[tau + n])]
            # print ( f"tau + n {tau+n}, T {T}, curr_state {curr_state}, rwd_state {rwd_state}, tau {tau}, n {n}, len_states {len(states)} len_rwd {len(rwds)}, G {G}")
            V_TD_error[states[tau]] += alpha * (G-V_TD_error[states[tau]])
        if tau == T-1:
            break
        t += 1
        # print (states)
        # print (rwds)
        # print (V_TD_error)
print (V_TD_error)

# the 2nd part of the question doesnt make sense as we calculated the same exact thing here
# even if the Vt is different it will sum up to the same as what we have here.