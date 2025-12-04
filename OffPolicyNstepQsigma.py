import numpy as np

np.random.seed(9)
stateA = (0,1)
stateB = (0,3)

actions = {0:np.array((-1,0)), 1:np.array((1,0)), 2: np.array((0,-1)), 3:np.array((0,1))}

n = 2
gamma = 0.99
alpha = 0.01
sigma = 0.5

Q = np.random.randn(4, 5, 5)
# print(Q.shape)
Q[3, 0, 0] += 10
Q[0, 1, 2] += 10
Q[2, 0, 2] += 10
Q[3, 0, 2] += 5
Q[0, 1, 3] += 5
Q[2, 0, 4] += 5

# Q2 = np.random.randn(4, 5, 5)
# Q2[3, 0, 0] += 10
# Q2[0, 1, 2] += 10
# Q2[2, 0, 2] += 10
# Q2[3, 0, 2] += 5
# Q2[0, 1, 3] += 5
# Q2[2, 0, 4] += 5

def action_to_idx(act):
    action_to_idx = -1
    if act[0] == -1:
        action_to_idx = 0
    elif act[0] == 1:
        action_to_idx = 1
    elif act[1] == -1:
        action_to_idx = 2
    elif act[1] == 1:
        action_to_idx = 3
    return action_to_idx

end_state = np.random.randint(0, 5, 2)
for iters in range(1000):
    print (f"\riters - {iters}", flush=True, end="")
    start_state = np.random.randint(0, 5, 2)

    if all(start_state == end_state):
        continue

    T = np.inf
    t = 0
    act = actions[np.random.choice(list(actions))]
    states = [start_state]
    acts = [act]
    rewards = [0]
    ro = [1]
    new_state = start_state
    # print (new_state, end_state)
    while True:
        if t<T:
            # print ("line59 t", t, "T", T, "new state", new_state, end_state)
            if all(new_state == end_state):
                # print ("reached terminal state - append 0 rwd and terminal state")
                new_state = end_state 
                T = t+1
                rewards.append(0)
                states.append(new_state)
            else:
                # print (f"new state {new_state}, stateA {stateA}, stateB {stateB}, action {act}")
                if all(new_state == stateA):
                    # print("equal A")
                    new_state = np.array((4,1))
                    rewards.append(10)
                    states.append(new_state)
                elif all(new_state == stateB):
                    # print("equal B")
                    new_state = np.array((3,3))
                    rewards.append(5)
                    states.append(new_state)
                #elif (new_state[0] + act[0] > 4) or (new_state[1] + act[1] > 4) or (new_state[0] + act[0] < 0) or (new_state[1] +act [1]< 0):
                elif any(new_state + act > 4) or any(new_state + act < 0) :
                    # print ("going out")
                    rewards.append(-1)
                    states.append(new_state)
                else:
                    new_state += act
                    # print (f"append {new_state}")
                    rewards.append(-1)
                    states.append(new_state)
                act = actions[np.random.choice(list(actions))]
                # print (f"action {act}")
                acts.append(act)
                max_per_state = np.max(Q, axis=0) #if np.random.random() < 0.5 else np.max(Q2, axis=0)
                is_max = (Q == max_per_state)
                ro.append(np.sum(is_max[action_to_idx(acts[t+1])][states[t+1][0]][states[t+1][1]])/4)
        tau = t - n + 1
        # print ("tau", tau, "T", T, "t", t)
        if tau >= 0:
            G = 0
            for k in range(min(t + 1, T), tau+1):
                if k == T:
                    G = rewards[T]
                else:
                    VHat = 0
                    for i in range(4):
                        VHat += is_max[i][states[k][0]][states[k][1]] * Q[i][states[k][0]][states[k][1]]/4.0
                    G = rewards[k] + gamma * (sigma * ro[k] + (1-sigma)*is_max[action_to_idx(acts[k])][states[k][0]][states[k][1]]/4) * (G - Q[action_to_idx(acts[k])][states[k][0]][states[k][1]]) + gamma*VHat
            Q[action_to_idx(acts[tau])][states[tau][0]][states[tau][1]] += alpha * (G - Q[action_to_idx(acts[tau])][states[tau][0]][states[tau][1]])
            if tau == T - 1:
                break
        t+=1
print ("\n")
print (np.argmax(Q, axis=0))
np.save("QryoffPolicy.npy", Q)