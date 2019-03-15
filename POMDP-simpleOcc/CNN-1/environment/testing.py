import numpy as np
import car
import random
import matplotlib.pyplot as plt
import time

import agent


optimize = agent.lateral_control()
y = 0
result = []
time = []
steering = []
steer = 0
for s in range(0,100):
    Vec, dVec, ddVec = optimize.quintic(s)



    params, t = optimize.solve(0,4,0,100)
    steer += optimize.steering(0,4,0,100,s)
    steering.append(steer)
    time.append(t)
    result.append(np.dot(params,Vec))

print(params)
print(float(sum(time)/len(time)))

plt.figure(1)
plt.plot(result)
plt.show()

plt.figure(2)
plt.plot(steering)
plt.show()



print(Vec.shape)
print(dVec.shape)
print(dVec.shape)

