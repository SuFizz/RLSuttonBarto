import numpy as np

grid = np.zeros((7,10))
grid[:,3:6] = 1
grid[:,6:8] = 2
grid[:,8] = 1
print (grid)

start = (3,0)
goal = (3, 7)
Q = np.zeros((8, 7, 10))
alpha = 0.5
epsilon = 0.1
iters = 0

actions = {0:(-1,-1), 1:(-1,0), 2:(-1,1), 3:(0,-1), 4:(0,1), 5:(1,-1), 6:(1,0), 7:(1,1)}
#actions = {0:(-1,0), 1:(0,-1), 2:(0,1), 3:(1,0)}

def take_action(state, act):
    new_state = (int(state[0]+actions[act][0]-grid[*state]), int(state[1]+actions[act][1]))
    #print (state, act, grid[*state], new_state)
    if (new_state[0] < 0 or new_state[0] > 6) and (new_state[1]<0 or new_state[1]>9):
        return state 
    if new_state[0] < 0 or new_state[0]>6:
        return (state[0], new_state[1])
    elif new_state[1] < 0 or new_state[1] > 9:
        return (new_state[0], state[1])
    else:
        return new_state

while iters < 10000:
    state = start
    act = np.argmax(Q, axis=0)[state] if np.random.random() > epsilon else np.random.choice(list(actions.keys()))
    iterat = 0
    state_not_varying = 0
    while True:
        # if iterat%1000==0:
        # print (f"\r iterat {iterat}", end="", flush=True)
        iterat+=1
        new_state = take_action(state, act)
        # print (new_state, act)
        new_act = np.argmax(Q, axis=0)[new_state] if np.random.random() > epsilon else np.random.choice(list(actions.keys()))
        rwd = -1 if new_state!=goal else 0
        Q[act, *state] = Q[act, *state] + alpha * (rwd + Q[new_act, *new_state] - Q[act, *state])
        if (new_state == state):
            state_not_varying+=1
        else:
            state_not_varying = 0
        state = new_state
        act = new_act
        if state == goal or state_not_varying > 50:
            break
    iters+=1
    print (f"\r iters {iters} iterat {iterat}", flush=True, end="")