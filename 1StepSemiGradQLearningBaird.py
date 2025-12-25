import jax
import jax.numpy as jnp
#Bairds counter example
max_states = 7
max_actions = 7

states = jnp.zeros(max_states)
on_policy_action = max_states - 1
gamma = 0.99
alpha = 0.1
n = 1

sub_key = jax.random.key(42)

def get_features(state, action):
    if state == max_states-1:
        return jnp.zeros(8)
    if action == max_actions-1:
        return jnp.array([0, 0, 0, 0, 0, 0, 1, 2])
    else:
        feature = jnp.zeros(8)
        feature = feature.at[state].set(2)
        feature = feature.at[max_states].set(1)
        return feature

def q_value(state, action, weights):
    features = get_features(state, action)
    return jnp.dot(features, weights)

def pi (action, state):
    if action == max_states - 1:
        return 1

weights = jnp.zeros(max_states+1)
print ("Start")
for episode in range(1):
    new_key, sub_key = jax.random.split(sub_key)
    states = jnp.array(jax.random.randint(sub_key, (), 0, max_states-1)) # because state 7 is terminal
    sigma = jnp.array(0.5)
    ro = jnp.array(max_states)
    
    #rewards are anyways 0 all through
    t = 1
    T = jnp.inf
    while True:
        if t<T:
            states = jnp.append(states, max_states-1)
            # print(states, states[-1])
            if states[-1] == max_states - 1:
                T = t + 1
            else:
                new_key, sub_key = jax.random.split(sub_key)
                actions = jnp.append(actions, jnp.array(jax.random.randint(sub_key, (), 0, max_states)))
                sigma = jnp.append(sigma, 0.5)
                ro = jnp.append(ro, 7)
        tau = t - n + 1
        # print ("tau", tau, "min", min(t+1, T))
        if tau >= 0:
            G = 0
            for k in range(min(t+1, T), tau, -1):
                print("k", k, "T", T, t)
                if k == T and t == 1000:
                    G = 0
                else:
                    v_bar = q_value(states[k], max_actions-1, weights) # policy pi always goes to the terminal state and then goes round and round in circles there
                    print(gamma, states[k], weights, pi(6, states[k]), q_value(states[k], 6, weights), v_bar, gamma)
                    G = 0 + gamma * (0.5 * 7 + (1 - 0.5)*pi(6, states[k]))*(G - q_value(states[k], 6, weights)) + gamma*v_bar
                    print(f"G {G}")
            weights = weights + alpha * (G - q_value(states[tau], max_actions-1, weights))*get_features(states[tau], max_actions-1)
        # if tau == T-1:
        #     break
        t = t+1
        if t > 1000:
            break
    print (f"\r{jnp.sum(jnp.abs(weights))} at episode-{episode}", flush=True, end="")
    if jnp.sum(weights) > 10e5:
        print ("Diverged at episode", episode)
        break