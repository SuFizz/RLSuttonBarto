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
    # if state == max_states-1:
    #     return jnp.zeros(8)
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

weights = jax.random.randint(sub_key, (8,), 0, max_states+1)
print ("Start")
for episode in range(3000):
    new_key, sub_key = jax.random.split(sub_key)
    present_states = jax.random.randint(sub_key, (), 0, max_states-1) # because state 6 is terminal
    next_state = max_states - 1
    max_qsa = 0
    for act in range(max_actions):
        qval = q_value(next_state, act, weights)
        if max_qsa < qval:
            max_qsa = qval
    # print ("\n",alpha, gamma, max_qsa, q_value(present_states, max_actions-1, weights), get_features(present_states, max_actions-1))
    weights += alpha * (0 + gamma*max_qsa - q_value(present_states, max_actions-1, weights))*get_features(present_states, max_actions-1)
    # print (weights)
    summer = jnp.sum(jnp.abs(weights))
    print (f"\rsum Weights {summer} at {episode}", end="", flush=True)
    if jnp.sum(jnp.abs(weights)) > 1e5:
        print(f"Divergence at {episode}")
        break