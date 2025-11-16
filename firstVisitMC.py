import numpy as np
import matplotlib.pyplot as plt
V = np.zeros((2, 10, 10)) # usability of ace, player cards sum, dealers showing card
returns = [[[[] for _ in range(10)] for _ in range(10)] for _ in range(2)]

player_start_sum = 12
iters = 0

# start from non usable ace

while iters < 500001:
    iters += 1
    if True and iters % 1000 == 0:
        print (f"\riter {iters}", end='', flush=True)
    used = np.zeros((2,10,10))
    G = 0
    #generate episode - hit until you get to either 20 or 21 and then stick
    dc0 = np.random.randint(1,13)
    dc0 = 10 if dc0 > 10 else dc0
    dc1 = np.random.randint(1,13)
    dc1 = 10 if dc1 > 10 else dc1
    dealer_sum = dc0 + dc1

    pc0 = np.random.randint(1,13)
    pc0 = min(10, pc0)
    pc1 = np.random.randint(1,13)
    pc1 = min(10, pc1)
    player_sum = pc0 + pc1

    end_of_episode = False

    # print("first", dc0, dc1, pc0, pc1, end_of_episode)

    # assume dc0 is dealer showing card
    while player_sum < 20:
        pc_new = np.random.randint(1,13)
        pc_new = min(10,pc_new) 
        #print ("ps-" , player_sum)
        player_sum += pc_new
        #print ("start1", player_sum, pc_new, dc0, dc1)
        if player_sum-pc_new >= player_start_sum:
            # print ("new player_sum : ", player_sum)
            if player_sum == 21:
                G += 1
                end_of_episode = True
            elif player_sum > 21:
                G += -1
                end_of_episode = True
            if used[0][player_sum - player_start_sum - pc_new][dc0 - 1]:
                print ("used this state:" , player_sum -player_start_sum - pc_new, dc0-1)
                assert 1
            else: 
                #print ("using state1 : " , player_sum-player_start_sum-pc_new, dc0-1, dc1-1, G)
                assert G <= 1 and G >= -1
                returns[0][player_sum - player_start_sum - pc_new][dc0 - 1].append(G)
                used[0][player_sum-player_start_sum-pc_new][dc0 - 1] = 1
                if dc1 != dc0 and not used[0][player_sum - player_start_sum - pc_new][dc1-1]:
                    #print ("updated dc1 - ", dc1)
                    returns[0][player_sum - player_start_sum - pc_new][dc1-1].append(G)
                    used[0][player_sum-player_start_sum-pc_new][dc1-1] = 1

    if end_of_episode:
        continue

    #print ("dealer now")
    while dealer_sum < 18:
        dc_new = np.random.randint(1,13)    
        dc_new = 10 if dc_new > 10 else dc_new
        dealer_sum += dc_new
        #print ("ds-", dealer_sum)
        if dealer_sum > 21 and not used[0][player_sum-player_start_sum][dc0 - 1]:
            returns[0][player_sum-player_start_sum][dc0 - 1].append(1)
            #print ("using state2 : " , player_sum-player_start_sum, dc0-1, dc1-1, 1)
            used[0][player_sum-player_start_sum][dc0 - 1] = 1
            end_of_episode = True
            if dc0 != dc1 and not used[0][player_sum-player_start_sum][dc1 - 1]:
                returns[0][player_sum-player_start_sum][dc1-1].append(1)
                used[0][player_sum-player_start_sum][dc1-1] = 1
    
    if end_of_episode:
        continue

    if player_sum < 21:
        if player_sum > dealer_sum: 
            G += 1
        elif dealer_sum > player_sum:
            G += -1
        elif dealer_sum == player_sum:
            G += 0

    assert G >=-1 and G <= 1
    #print ("using state3 : " , player_sum-player_start_sum, dc0-1, dc1-1, G)
    if not used[0][player_sum - player_start_sum][dc0-1]:
        returns[0][player_sum - player_start_sum][dc0 - 1].append(G)
    if dc0 != dc1 and not used[0][player_sum-player_start_sum][dc1 - 1]:
        returns[0][player_sum - player_start_sum][dc1-1].append(G)

    #end of episode

for i in range(10):
    for j in range(10):
        V[0][i][j] += sum(returns[0][i][j])/(1e-9 + len(returns[0][i][j]))

print ("\n",V[0])

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X,Y = np.meshgrid(np.arange(10), np.arange(10))

# Plot the surface
ax.plot_surface(X,Y, V[0], cmap='terrain', edgecolor ='none')

# Add labels and title
ax.set_xlabel('player sum')
ax.set_ylabel('dealer shows')
ax.set_zlabel('State value function')
ax.set_title('50000 iterations')

# Display the plot
plt.show()