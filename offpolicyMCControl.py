import numpy as np

race_track = np.array([np.arange(50)]*50)
np.random.seed(100)
z = np.random.randint(0,30, (50,))
zeroer = z.reshape(-1,1)
race_track = race_track - zeroer 
race_track[race_track<0] = 0
race_track[race_track>0] = 1
np.set_printoptions(threshold=np.inf, linewidth=200)

def action_to_index(act):
    return act[-1]*3 + act[1]

def index_to_action(idx):
    return (idx//2-1, idx%3-1)

for i in range(10,50):
    race_track[i][np.random.randint(40,50):] = 0

Q = np.random.uniform(0,1,(9, 50, 50, 6, 6)) # actions, racetrack_y, racetrack_x, vel_y, vel_x --> Assuming y goes downward
for action_idx in range(9):
    ay, ax = index_to_action(action_idx)
    # Reward actions that accelerate toward finish line (positive x direction)
    if ax > 0:  # Moving right toward x=49
        Q[action_idx] = 5.0
    elif ax == 0:
        Q[action_idx] = 2.0
    else:  # ax < 0, moving away from goal
        Q[action_idx] = -2.0
actions = [-1, 0, 1]
C = np.zeros_like(Q)
pi = np.argmax(Q, axis=0)
iters = 0

print (race_track)

def get_random_start():
    nonzero_indices = np.argwhere(race_track[49] > 0)
    random_idx_in_list = np.random.choice(len(nonzero_indices))
    random_start = tuple (nonzero_indices[random_idx_in_list])
    return random_start[0]

def generate_episode():
    S = []
    A = []
    R = []
    while True:
        velocity = [0,0]
        action  = np.array((np.random.choice(actions[1:]), np.random.choice(actions[1:])))
        while True:
            action  = np.array((np.random.choice(actions[1:]), np.random.choice(actions[1:])))
            if action[0] or action[1]:
                break
        velocity += action
        S.append(np.array((49,get_random_start(), 0, 0)))
        A.append(action)
        #print("state1", S[-1].tolist())
        #print("act1", A[-1].tolist())
        while True:
            ##print ("st_vel",S[-1][0], velocity[0], S[-1][1], velocity[1])
            new_state = np.array((S[-1][0] - velocity[0], S[-1][1] + velocity[1], velocity[0], velocity[1]))
            ##print (new_state)
            y_at_50 = S[-1][0] if velocity[0] == 0 else  (50 - S[-1][1]) * (float(velocity[1]/velocity[0])) + S[-1][0] # check if trajectory crosses the finish line
            if y_at_50 < 10 and y_at_50 >= 0 and new_state[1] >= 49:
                R.append(1)
                #print("rwdi0 given", len(R), len(S), len(A))
                assert len(R) == len(S)
                assert len(S) == len(A)
                #print("end-ep")
                return S, A, R

            R.append(-1)
            #print("rwd-1 given")
            if new_state[0] < 0 or new_state[0] >= 50:
                #print ("break1")
                break
            if new_state[1] < 0 or new_state[1] >= 50:
                #print ("break2")
                break
            #S[-1] to new_state
            
            min_y = min(new_state[0], S[-1][0])
            max_y = max(new_state[0], S[-1][0])

            x2 = new_state[1]
            x1 = S[-1][1]
            y2 = new_state[0]
            y1 = S[-1][0]
            if np.abs(x2 - x1) > 1 and np.abs(y2- y1) > 1:
                c = (x2*y1 - x1*y2)/float(x2 - x1)
                m = (y2 - y1)/float(x2 - x1)
                #print (f"({y2},{x2}); ({y1, x1})")

                jumped = False
                for y_alpha in range(min_y+1, max_y):
                    x_alpha = (y_alpha - c)/m
                    #print ("y_alpha, x_alpha", y_alpha, x_alpha)
                    if not race_track[y_alpha][int(x_alpha)] or not race_track[y_alpha][int(x_alpha)+1]:
                        jumped = True
                        #print("jumping across fence")
                        break
                if jumped:
                    break

            if race_track[new_state[0]][new_state[1]]:
                S.append(new_state)
                #print ("state app-", new_state)

                act = np.zeros(2)
                while True:
                    tmp_vel = velocity.copy()
                    act  = np.array((np.random.choice(actions), np.random.choice(actions)))
                    if new_state[0] < 10:
                        act = np.array((0,1)) if velocity[0] <=0 else np.array((-1, 1))
                        vel = tmp_vel + act
                        if vel[1] > 5:
                            act[1] = 0
                        #print (new_state, act, tmp_vel)
                    ##print("prev-",tmp_vel)
                    tmp_vel += act
                    ##print("next-", act, tmp_vel)
                    if any(tmp_vel) and all(tmp_vel>=0) and all(tmp_vel < 6):
                        break
                velocity += act
                A.append(act)
                ##print ("mid-ep")
                #print("action:", A[-1].tolist())
                #print ("continue2")
                continue
            else:
                #print ("break3")
                break

while iters < 10000:
    if iters % 100 == 1:
        print (f"\riteration {iters}", flush = True, end='')
    iters += 1
    S, A, R = generate_episode()
    G = 0
    W = 1
    iters_sof = 1
    for step in reversed(range(len(S))):
        G += R[step]
        act = A[step]
        state_act_pair = tuple((action_to_index(act+1), S[step][0], S[step][1], S[step][2], S[step][3]))
        #print (action_to_index(act+1), S[step][0], S[step][1], S[step][2], S[step][3] )
        C[state_act_pair] += W
        #print (C[state_act_pair])
        old_q = Q[state_act_pair]
        print (f" W {W}, C {C[state_act_pair]}, G {G}, Q {Q[state_act_pair]}, R {R[step]}")
        Q[state_act_pair] += float(W)*(float(G) - float(Q[state_act_pair]))/float(C[state_act_pair])
        print (tuple(S[step]))
        #print (np.argmax(Q, axis=0))
        #print (action_to_index(act+1))
        #print (S[step], np.argmax(Q, axis=0)[S[step]], action_to_index(act+1))
        #print ("delta q", Q[state_act_pair] - old_q)
        
        if np.argmax(Q, axis=0)[tuple(S[step])] != action_to_index(act+1):
            print (np.argmax(Q, axis=0)[tuple(S[step])], action_to_index(act+1))
            break
        print (f"\riter {iters_sof}", end='', flush=True)
        iters_sof += 1
        W = W*9
#print (np.argmax(Q, axis=0))
np.save("Qarray.npy", Q)