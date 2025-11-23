import numpy as np

race_track = np.array([np.arange(50)]*50)
np.random.seed(100)
z = np.random.randint(0,30, (50,))
zeroer = z.reshape(-1,1)
race_track = race_track - zeroer 
race_track[race_track<0] = 0
race_track[race_track>0] = 1
np.set_printoptions(threshold=np.inf, linewidth=200)

for i in range(10,50):
    race_track[i][np.random.randint(40,50):] = 0

Q = np.random.randn(9, 50, 50, 6, 6) # actions, racetrack_y, racetrack_x, vel_y, vel_x --> Assuming y goes downward
actions = [-1, 0, 1]
C = np.zeros_like(Q)
pi = np.argmax(Q, axis=0)
iters = 0

print (race_track)
Q = np.load("Qarray.npy")
i = 0

def action_to_index(act):
    return act[0]*3 + act[1]

def index_to_action(idx):
    return (idx//3-1, idx%3-1)

def get_random_start():
    nonzero_indices = np.argwhere(race_track[49] > 0)
    random_idx_in_list = np.random.choice(len(nonzero_indices))
    random_start = tuple (nonzero_indices[random_idx_in_list])
    return 30#random_start[0]

state = np.array((49,get_random_start(),0,0))
velocity = (0,0)

qargmax = np.argmax(Q, axis=0)
while i < 100:
    print(state)
    next_action = qargmax[tuple(state)]
    action_pair = index_to_action(next_action)
    velocity += action_pair
    print (next_action)
    print (action_pair)
    #print (velocity)

    state = np.array((state[0]-velocity[0], state[1]+velocity[1], velocity[0], velocity[1]))
    i += 1