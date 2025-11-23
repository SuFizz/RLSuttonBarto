import numpy as np
import matplotlib.pyplot as plt

V = np.zeros(101, dtype=np.float32)
action_space = np.zeros(101, dtype = np.int16)

gamma = 1 
prob_h = 0.4
iter = 0

while True:
    delta = 0
    for s in range(1, 100):
        V_old = V[s]
        v_max = -np.inf
        for act in range(1, min(s, 100-s)+1):
            # you can either win or lose and those are the only new states you can get to
            vst_a = ((1-prob_h) * (gamma*V[s-act]) + (prob_h) * (1+gamma*V[s+act])) if s+act == 100 else ((1-prob_h) * (gamma*V[s-act]) + (prob_h) * (gamma*V[s+act]))
            if s+act>100:
                print(V[s+act])
            #print (s, act, V_old, vst_a)
            #print (act, s, iter, vst_a, v_max)
            if vst_a > v_max:
                if (vst_a > 1):
                    exit(-1)
                v_max = vst_a
        delta = max(delta, abs(V_old-v_max))
        V[s] = v_max
    iter += 1
    print (f"Iteration {iter} delta : {delta} V_old : {V_old} Vs : {V[s]}")
    if delta == 0:
        break

for s in range(1, 100):
    V_old = V[s]
    v_max = -np.inf
    for act in range(1, min(s, 100-s)+1):
        vst_a = 0
        # you can either win or lose and those are the only new states you can get to
        vst_a += ((1-prob_h) * (gamma*V[s-act]) + (prob_h) * (1+gamma*V[s+act])) if s+act == 100 else ((1-prob_h) * (gamma*V[s-act]) + (prob_h) * (gamma*V[s+act]))
        #print (s, act, V_old, vst_a)
        if vst_a > v_max:
            v_max = vst_a
            action_space[s] = act

print(action_space)
plt.plot(action_space)
#plt.plot(V)
plt.show()