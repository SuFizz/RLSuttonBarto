import numpy as np
# import matplotlib.pyplot as plt


starting_state = np.array([0, 0])
rental_lambda = (3,4)
return_lambda = (3,2)

gamma = 0.9
#V = np.random.uniform(0, 1, (21,21))
V = np.full((21, 21), 700)

actions = np.arange(-5,6)
action_space = np.zeros((21,21))

# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots()

# im = ax.imshow(action_space, cmap='viridis', interpolation='nearest')
# # Add a colorbar
# plt.colorbar(im, label='action', ax=ax)
# ax.set_title('Policy Iteration 0')
# ax.set_xlabel('Cars in loc2')
# ax.set_ylabel('Cars in loc1')


# Define the range of numbers
numbers = np.arange(21)  # Numbers from 0 to 20
x, y = np.meshgrid(numbers, numbers)
cars_available = np.stack((x.ravel(), y.ravel()), axis=-1)

lambda_3 = np.linspace(0, 7, 8, dtype=int)
lambda_2 = np.linspace(0, 5, 6, dtype=int)
lambda_4 = np.linspace(0, 9, 10, dtype=int)

factorials = np.ones(10)
for f in range(1, 10):
    factorials[f] = factorials[f-1] * f

poisson_2 = [(2**a)*(np.exp(-2))/factorials[a] for a in lambda_2]
poisson_3 = [(3**a)*(np.exp(-3))/factorials[a] for a in lambda_3]
poisson_4 = [(4**a)*(np.exp(-4))/factorials[a] for a in lambda_4]

x_rental, y_rental = np.meshgrid(poisson_3, poisson_4)
x_return, y_return = np.meshgrid(poisson_3, poisson_2)
rentals = np.stack((x_rental.ravel(), y_rental.ravel()), axis=-1)
returns = np.stack((x_return.ravel(), y_return.ravel()), axis=-1)

x_rental_nums, y_rental_nums = np.meshgrid(lambda_3, lambda_4)
x_return_nums, y_return_nums = np.meshgrid(lambda_3, lambda_2)
rental_nums = np.stack((x_rental_nums.ravel(), y_rental_nums.ravel()), axis=-1)
return_nums = np.stack((x_return_nums.ravel(), y_return_nums.ravel()), axis=-1)

assert (len(rental_nums) == len(rentals))
assert (len(return_nums) == len(returns))

theta = 1

while True:
    pol_iter = 1
    # each i,j correspond to first, second locations
    iter = 1
    print (f"Policy Iter : {pol_iter} ")
    while True:
        delta = 0
        for cars_loc1,cars_loc2 in cars_available:
            new_v = 0
            for rnt in range(len(rentals)):
                for ret in range(len(returns)):
                    rent1, rent2 = rentals[rnt] # probabilities in either of the locations
                    rent_num1, rent_num2 = rental_nums[rnt] # number of rentals corresponding to above

                    return1, return2 = returns[ret] # probabilities of returns in either of the locations
                    return_num1, return_num2 = return_nums[ret] # number of returns corresponding to above probabilities
                    # print (cars_loc1, rent_num1, cars_loc2, rent_num2, return_num1, return_num2)
                    actual_rented1, actual_rented2 = min(rent_num1, cars_loc1), min(rent_num2, cars_loc2)

                    reward = 10*(actual_rented1 + actual_rented2) 
                    new_state_1 = (cars_loc1 - actual_rented1 + return_num1) if ((cars_loc1 - actual_rented1 + return_num1) < 20) else 20
                    new_state_2 = (cars_loc2 - actual_rented2 + return_num2) if ((cars_loc2 - actual_rented2 + return_num2) < 20) else 20
                    # print (new_state_1, new_state_2)
                    new_v += rent1 * rent2 * return1 * return2 * (reward + gamma * V[new_state_1][new_state_2]) 
            delta = max(delta, np.abs(V[cars_loc1][cars_loc2] - new_v))
            V[cars_loc1][cars_loc2] = new_v
        print (f"\rPolicy iter : {pol_iter} iteration num : {iter} delta : {delta}", end = '', flush=True)
        iter += 1
        if delta < theta:
            break

    old_action_space = action_space.copy()

    for cars_loc1, cars_loc2 in cars_available:
        max_rwd = 0
        for act in actions:
            act_state_1, act_state_2 = cars_loc1 - act, cars_loc2 + act
            if act_state_1 < 0 or act_state_1 > 20 or act_state_2 < 0 or act_state_2 > 20:
                continue    #actual state will come from going to a different point
            vst = -2 * np.abs(act) + V[act_state_1][act_state_2]
            if vst > max_rwd:
                max_rwd = vst
                action_space[cars_loc1][cars_loc2] = act
    # im.set_data(action_space)
    # ax.set_title(f"Policy Iteration {pol_iter}")
    # plt.pause(2)
    # Plot the heatmap
    if np.array_equal(action_space, old_action_space):
        print(action_space, old_action_space)
        break
    pol_iter+= 1
# plt.ioff()
# plt.show()