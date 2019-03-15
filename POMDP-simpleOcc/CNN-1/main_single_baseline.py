import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#### Import own modules ####
import sys
sys.path.insert(0,'environment/')
import q_learning
import world
import lateral_agent
import car

#### Environment parameters ####

num_of_cars = 5
num_of_lanes = 2
track_length = 300
speed_limit = 120
random_seed = 0
random.seed(random_seed)
x_range = 10


#### Ego parameters ####

ego_lane_init = 1
ego_pos_init = 0
ego_speed_init = speed_limit

#### Network parameters ####

input_dim = (num_of_cars+1)*3
output_dim = x_range*num_of_lanes
hidden_units = 100
layers = 5
clip_value = 300
learning_rate = 0.001
buffer_size = 50000
batch_size = 32
update_freq = 6000

#### RL Parameters ####

gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 100000

#### Learning Parameters ####

max_train_episodes = 10000
pre_train_steps = 10000
random_sweep = 3
tau = 1


#### Environment ####

done = False
dt = 0.05
timestep = 0

lateral_controller = lateral_agent.lateral_control(dt)
env = world.World(num_of_cars,num_of_lanes,track_length,speed_limit,ego_pos_init,ego_lane_init,ego_speed_init,dt,random_seed,x_range)
goal_lane = (ego_lane_init - 1) * env.road_width + env.road_width * 0.5
goal_lane_prev = goal_lane
action = np.zeros(1) # acc/steer split up

#### Plot variables ####
max_timestep = 1250
x_ego_list = np.zeros((random_sweep,max_timestep))
y_ego_list = np.zeros((random_sweep,max_timestep))
y_acc_list = np.zeros((random_sweep,max_timestep))
v_ego_list = np.zeros((random_sweep,max_timestep))
x_acc_list = np.zeros((random_sweep,max_timestep))
reward_list = np.zeros((random_sweep,max_timestep))
reward_sum_list = np.zeros((random_sweep,max_timestep))


for r_seed in range(0,random_sweep):

    random.seed(r_seed)

    #### Start training process ####

    states = []
    actions = []
    reward_time = []

    folder_path = './initial_testing'

    path = folder_path+ "model_initial"


    done =False


    timestep = 0
    env = world.World(num_of_cars, num_of_lanes, track_length, speed_limit, ego_pos_init, ego_lane_init, ego_speed_init,
                      dt, r_seed, x_range)
    action = 0
    reward_sum = 0
    while done == False:

        #if timestep % 10 == 0:
        #    action = random.randint(0,num_of_lanes*x_range-1)
        #    action = 10

        if timestep  == 900:
            action = random.randint(0,num_of_lanes*x_range-1)
            action = 2
            #action = 0

        if timestep  == 50:
            action = random.randint(0,num_of_lanes*x_range-1)
            action = 0
            action = 12


        #### Track parameters
        _, reward_list[r_seed, timestep], done = env.step(action)
        x_acc_list[r_seed, timestep], y_acc_list[r_seed, timestep] = env.return_acc()
        x_ego_list[r_seed, timestep], _, _, _, v_ego_list[r_seed, timestep] = env.get_ego()


        reward_sum += reward_list[r_seed, timestep]
        reward_sum_list[r_seed,timestep] = reward_sum
        #env.render()
        timestep += 1





plt.figure(1)
ax1 = plt.subplot(3,1,1)
#ax1.plot(x_ego_list,y_acc_list,'x')
sns.tsplot(y_acc_list)
ax1.set_xlabel('timestep')
ax1.set_ylabel('acc in [m/s^2]')
ax1.set_title('y-acceleration')
ax2 = plt.subplot(3,1,2)
#ax2.plot(x_ego_list,v_ego_list,'x')
sns.tsplot(v_ego_list)
ax2.set_xlabel('timestep')
ax2.set_ylabel('velocity in [m/s]')
ax2.set_title('Velocity')
ax3 = plt.subplot(3,1,3)
#ax3.plot(x_ego_list,x_acc_list,'x')
sns.tsplot(x_acc_list)
ax3.set_xlabel('timestep')
ax3.set_ylabel('acc in [m/s^2]')
ax3.set_title('x-acceleration')
plt.tight_layout()
plt.show(block=False)


plt.figure(2)
ax1 = plt.subplot(2,1,1)
#ax1.plot(x_ego_list,y_acc_list,'x')
sns.tsplot(reward_list)
ax1.set_xlabel('timestep')
ax1.set_ylabel('reward')
ax1.set_title('Reward')
ax2 = plt.subplot(2,1,2)
#ax1.plot(x_ego_list,y_acc_list,'x')
sns.tsplot(reward_sum_list)
ax2.set_xlabel('timestep')
ax2.set_ylabel('sum of rewards')
ax2.set_title('Reward sum')
plt.tight_layout()

plt.show(block = False)


plt.figure(3)
ax1 = plt.subplot(1,1,1)
#ax1.plot(x_ego_list,y_acc_list,'x')
sns.tsplot(x_ego_list)
ax1.set_xlabel('timestep')
ax1.set_ylabel('x-distance in [m]')
ax1.set_title('Distance traveled')
plt.show()




