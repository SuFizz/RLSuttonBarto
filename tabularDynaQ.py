import numpy as np
import matplotlib.pyplot as plt

maze = np.zeros((6,9))
maze[1:4,2] = 1
maze[4,5] = 1
maze[:3,-2] = 1
start = np.array((2,0))
goal = np.array((0,8))
epsilon = 0.9
alpha = 0.1
gamma = 0.95

# model = [[[[None for _ in range(9)] for _ in range(6)] for _ in range(4)]]
model = {}
# print (model)
# states = []
# acts = []

num_steps_0 = []

actions = {0:np.array((-1,0)), 1:np.array((1,0)), 2: np.array((0,-1)), 3:np.array((0,1))}

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

np.random.seed(42)
n = 0
Q = np.random.randn(4,6,9)
# Q[3,:,:] = 10
# Q[0,:,8] = 20
for iters in range(1000):
    state = start.copy() 
    # while True:
    #     state = np.array((np.random.randint(0,5), np.random.randint(0,8)))
    #     if not maze[state[0]][state[1]]:
    #         break
    states = []
    acts = []
    num_sts= 0
    while True:
        action = np.argmax(Q, axis=0)[state[0]][state[1]] if np.random.random() < epsilon else np.random.randint(0,4)
        # states = [state]
        # acts = [action]
        states.append(state)
        acts.append(action)
        num_sts += 1
        # print (state, action)
        rwd = -1
        if all(state + actions[action] == goal):
            rwd = 1
        new_state = state + actions[action]
        # print (new_state, maze[new_state[0]][new_state[1]])
        if any(new_state < 0) or new_state[0] > 5 or new_state[1] > 8 or maze[new_state[0]][new_state[1]]:
            new_state = state.copy() 
        Q[action][state[0]][state[1]] += alpha * (rwd + gamma * np.max(Q, axis=0)[new_state[0]][new_state[1]] - Q[action][state[0]][state[1]])
        # print(state, action, rwd, new_state)
        # print (len(model), len(model[0]), len(model[0][0]))
        model[(action,tuple(state))] = (rwd, new_state)
        for i in range(n):
            choice = np.random.randint(0,len(states))
            st = states[choice]
            ac = acts[choice]
            rwd, new_state = model[(ac,tuple(st))]
            # print (st, actions[ac], rwd, new_state)
            Q[ac][st[0]][st[1]] += alpha * (rwd + np.max(Q, axis=0)[new_state[0]][new_state[1]] - Q[ac][st[0]][st[1]])
        state = new_state
        if all(state == goal):
            # print ("breaking", state, goal, num_sts)
            break
    
    num_steps = 0
    check_state = start.copy()
    while num_steps < 950:
        action = np.argmax(Q, axis = 0)[check_state[0]][check_state[1]] if np.random.random() < epsilon else np.random.randint(4)
        # print (check_state, actions[action])
        num_steps += 1
        if all(check_state + actions[action] == goal):
            break
        new_check_state = check_state + actions[action]
        if any(new_check_state < 0) or new_check_state[0] >=6  or new_check_state[1] >= 9 or maze[new_check_state[0]][new_check_state[1]]:
            new_check_state = check_state.copy()
        check_state = new_check_state
    
    num_steps_0.append(num_steps)
    print (num_steps)
plt.plot(num_steps_0)
plt.show()