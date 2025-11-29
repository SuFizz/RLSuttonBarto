import numpy as np

np.random.seed(9)
stateA = (0,1)
stateB = (0,3)

actions = {0:np.array((-1,0)), 1:np.array((1,0)), 2: np.array((0,-1)), 3:np.array((0,1))}

n = 2
gamma = 0.99
alpha = 0.01

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
        tau = t - n + 1
        # print ("tau", tau, "T", T, "t", t)
        if tau >= 0:
            max_per_state = np.max(Q, axis=0) #if np.random.random() < 0.5 else np.max(Q2, axis=0)
            is_max = (Q == max_per_state)
            ro = 1
            G = 0

            # max_per_state2 = np.max(Q2, axis=0)
            # is_max2 = (Q2 == max_per_state)
            # ro2 = 1 
            # G2 = 0
            for i in range(tau+1, 1+min(tau + n -1, T - 1)):
                ro *= np.sum(is_max[action_to_idx(acts[i])][states[i][0]][states[i][1]])/4
                # ro2 *= np.sum(is_max2[acts[i], tuple(states[i])])/4
            for i in range(tau+1, 1+min(tau+n,T)):
                # print (f"rwd-{i} {rewards[i]}")
                G += gamma**(i-tau-1)*rewards[i]
                # print ("gamma ",gamma, " i-tau-1", i-tau-1)#, ""rewards[i])
            # print("G",G)
            if tau + n < T:
                action_to_index = action_to_idx(acts[tau + n])
                # print ("act-",acts[tau + n], "state-",states[tau + n][0], states[tau + n][1])
                # # print("Q",Q)
                G += gamma**n * Q[action_to_index][states[tau + n][0]][states[tau+n][1]]
                Q[action_to_index][states[tau + n][0]][states[tau+n][1]] += alpha * ro * (G - Q[action_to_index][states[tau + n][0]][states[tau+n][1]])
                # G2 = G + gamma**n * Q2[acts[tau + n], tuple(states[tau + n])]
            elif tau + n > T:
                # print ("len state-", len(states), "tau + n", tau+n)
                Q[action_to_index][states[T][0]][states[T][1]] += alpha * ro * (G - Q[action_to_index][states[T][0]][states[T][1]])
            if tau == T - 1:
                break
        t+=1
print ("\n")
print (np.argmax(Q, axis=0))
np.save("QryoffPolicy.npy", Q)