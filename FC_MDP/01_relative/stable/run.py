import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.reset_orig()

#### Import own modules ####
import sys
sys.path.insert(0,'environment/')
import q_learning
import world
import lateral_agent
import car

def vectorize_state(state):
    v_x = []
    v_y = []
    v_v = []

    for id_n in range(len(state)):
        v_x.append(state[id_n].x)
        v_y.append(state[id_n].y)
        v_v.append(state[id_n].v)


    state_v = np.concatenate((v_x,v_y))
    state_v = np.concatenate((state_v, v_v))
    return state_v


def relative_state(state):

    for id_n in range(len(state)-1,-1,-1):
        state[id_n].x = state[id_n].x-state[0].x
        state[id_n].y = state[id_n].x - state[0].y
        state[id_n].v = state[id_n].x - state[0].v

    return state


#### Environment parameters ####

num_of_cars = 2
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
output_dim = 21
hidden_units = 50
layers = 3
clip_value = 5000
learning_rate = 0.001
buffer_size = 50000
batch_size = 32
update_freq = 10000

#### RL Parameters ####

gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 100000

#### Learning Parameters ####

max_train_episodes = 20000
pre_train_steps = 100000
random_sweep = 10
tau = 1


#### Environment ####

done = False
dt = 0.1
timestep = 0

lateral_controller = lateral_agent.lateral_control(dt)
env = world.World(num_of_cars,num_of_lanes,track_length,speed_limit,ego_pos_init,ego_lane_init,ego_speed_init,dt,random_seed,x_range)
goal_lane = (ego_lane_init - 1) * env.road_width + env.road_width * 0.5
goal_lane_prev = goal_lane
action = np.zeros(1) # acc/steer


tf.reset_default_graph()

mainQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)
targetQN = q_learning.qnetwork(input_dim,output_dim,hidden_units,layers,learning_rate,clip_value)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf. trainable_variables()

targetOps = q_learning.updateNetwork(trainables,tau)

random_sweep= 5

## Init environment ##

states = []
actions = []
reward_time = []
reward_average = []
reward_episode = 0
total_steps = 0

done = False
num_of_episodes = "final"
r_seed = 2

#final_save_path = "./long/model_long_random/modelRL_0_"+str(num_of_episodes)+ ".ckpt"
#final_save_path = "./models/stable_r0/random_0_Final.ckpt"
#final_save_path = "./training/testing_13/modelRL_"+str(r_seed)+"_"+str(num_of_episodes)+ ".ckpt"
final_save_path = "./training/testing_03/modelRL_"+str(r_seed)+"_"+str(num_of_episodes)+ ".ckpt"


# Plotting/Testing Envionment
max_timestep = 600
num_tries = 10
num_of_finished = 0

x_ego_list = np.zeros((num_tries,max_timestep))
y_ego_list = np.zeros((num_tries,max_timestep))
y_acc_list = np.zeros((num_tries,max_timestep))
v_ego_list = np.zeros((num_tries,max_timestep))
x_acc_list = np.zeros((num_tries,max_timestep))
reward_list = np.zeros((num_tries,max_timestep))
action_list = np.empty((num_tries,max_timestep))
action_list_2 = []
q_values_list = np.empty((num_tries,int(max_timestep/10)))




for t in range(0,num_tries):
    if t % 10 == 0:
        print("Number of tries:" + str(t))
    with tf.Session() as sess:
        done = False
        sess.run(init)
        saver.restore(sess,final_save_path)
        env = world.World(num_of_cars, num_of_lanes, track_length, speed_limit, ego_pos_init, ego_lane_init,
                          ego_speed_init, dt, random_seed, x_range)

        state,_,_ = env.get_state()
        state = relative_state(state)
        state_v = vectorize_state(state)
        rewards = []
        test = 0
        flag = 0
        timestep = 0
        total_reward = 0
        while done == False:

            if timestep % 5 == 0:

                action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state_v]})
                action = action[0]
                q_values = sess.run(mainQN.output_q_predict,feed_dict={mainQN.input_state:[state_v]})
                q_values_list[t, int(timestep/10)] = np.amax(q_values)

                #action = random.randint(0, x_range * num_of_lanes-1)
                #print("Action: ", action, "Timestep: ", timestep)
            #action = random.randint(0,x_range*num_of_lanes)

            state1,reward,done = env.step(action)
            state1 = relative_state(state1)
            rewards.append(reward)
            total_reward += reward
            reward_list[t,timestep] = total_reward
            #action_list[t, timestep] = action
            #action_list_2.append(action)

            env.render()
            if env.success == True:
                num_of_finished += 1

            ### Position
            x_ego_list[t, timestep] = env.vehicle_list[0].x
            y_ego_list[t, timestep] = env.vehicle_list[0].y
            ### Driving Params
            v_ego_list[t, timestep] = env.vehicle_list[0].v
            y_acc_list[t, timestep] = env.y_acc
            x_acc_list[t, timestep] = env.x_acc

            state1_v = vectorize_state(state1)
            state_v = state1_v
            timestep += 1

        reward_time.append(sum(rewards))





image_save_path = './plots/'

#### Add position Distribution
x_ego_list[x_ego_list==0] = np.nan
y_ego_list[y_ego_list==0] = np.nan
plt.figure(2)
ax1 = plt.subplot(1,1,1)
ax1.plot(x_ego_list.T,y_ego_list.T)
ax1.set_xlabel('x-position in [m]')
ax1.set_ylabel('y-position in [m]')
ax1.set_title('Trajectory distribution')
ax1.grid()
plt.savefig(image_save_path + str(num_of_episodes) + "_trajectory_10" +str(r_seed)+".png")
plt.show(block=False)




#### Add Driving parameter distribution

plt.figure(3)
ax1 = plt.subplot(3,1,1)
sns.tsplot(v_ego_list)
#sns.lineplot(x="timestep",y="velocity",data=v_ego_list)
ax1.set_xlabel('timestep')
ax1.set_ylabel('velocity in [m/s]')
ax1.set_title('Velocity')
ax2 = plt.subplot(3,1,2)
sns.tsplot(y_acc_list)
ax2.set_xlabel('timestep')
ax2.set_ylabel('acc in [m/s^2]')
ax2.set_title('y-acceleration')
ax3 = plt.subplot(3,1,3)
sns.tsplot(x_acc_list)
ax3.set_xlabel('timestep')
ax3.set_ylabel('acc in [m/s^2]')
ax3.set_title('x-acceleration')
plt.tight_layout()
plt.savefig(image_save_path + str(num_of_episodes) + "_behaviour_10"+str(r_seed)+".png")
plt.show(block=False)
#plt.show(block=False)

#### Add reward and finished

plt.figure(4)
ax1 = plt.subplot(1,1,1)
sns.tsplot(reward_list)
ax1.set_xlabel("timestep")
ax1.set_ylabel("reward")
ax1.set_title("Reward")
plt.savefig(image_save_path + str(num_of_episodes) + "_reward_10"+str(r_seed)+".png")
plt.show(block=False)
#plt.show()

plt.figure(5)
ax1 = plt.subplot(1,1,1)
sns.tsplot(q_values_list)
ax1.set_xlabel("timestep")
ax1.set_ylabel("Q-value")
ax1.set_title("Q-values")
plt.savefig(image_save_path + str(num_of_episodes) + "_Q_values_10" +str(r_seed)+".png")
#plt.show(block=False)
plt.show()



#plt.figure(5)
#ax1 = plt.subplot(1,1,1)
#ax1.hist(action_list, bins=20,range=(0,19),histtype='step')
#plt.show(block=False)

#plt.figure(6)
#ax1 = plt.subplot(1,1,1)
#ax1.hist(action_list_2, bins=20,range=(0,19),histtype='step')
#plt.show()




average_reward = sum(reward_time)/num_tries

print("Average reward over " + str(num_tries) + " try is:" + str(average_reward))
print("Number of successfully finished trials: ",num_of_finished)
