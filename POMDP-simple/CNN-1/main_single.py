import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

#### Import own modules ####
import sys
sys.path.insert(0,'environment/')
import q_learning_pomdp
import world_pomdp
import lateral_agent

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


def processState(states):
    return np.reshape(states,[input_dim_v])




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
x_view = 150
y_view = 10

#### Network parameters ####

input_dim = [x_view*2+1,y_view*2+1]
input_dim_v = (x_view*2+1)*(y_view*2+1)*8
output_dim = (x_range*num_of_lanes)
hidden_units = 100
layers = 4
clip_value = 7500
learning_rate = 0.001
buffer_size = 100000
batch_size = 32
update_freq = 20000
kernel_size = [2,2]
stride = [2,2]

#### RL Parameters ####

gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 1500000

#### Learning Parameters ####

max_train_episodes = 25000
pre_train_steps = 50000
random_sweep = 5
tau = 1


#### Environment ####

done = False
dt = 0.1
timestep = 0

lateral_controller = lateral_agent.lateral_control(dt)
env = world_pomdp.World(num_of_cars,num_of_lanes,track_length,speed_limit,ego_pos_init,ego_lane_init,ego_speed_init,dt,random_seed,x_range)
goal_lane = (ego_lane_init - 1) * env.road_width + env.road_width * 0.5
goal_lane_prev = goal_lane
action = np.zeros(1) # acc/steer

#### Plot variables ####
max_timestep = 750
average_window = 100
finished = 0
x_ego_list = np.zeros((random_sweep,max_timestep))
y_ego_list = np.zeros((random_sweep,max_timestep))
y_acc_list = np.zeros((random_sweep,max_timestep))
v_ego_list = np.zeros((random_sweep,max_timestep))
x_acc_list = np.zeros((random_sweep,max_timestep))
reward_list = np.zeros((random_sweep,max_timestep))
reward_sum_list = np.zeros((random_sweep,max_train_episodes))
reward_average = np.zeros((random_sweep,int(max_train_episodes/average_window)))
finished_average = np.zeros((random_sweep,int(max_train_episodes/average_window)))

param_id = "test"

for r_seed in range(0,random_sweep):
    start = time.time()

    random.seed(r_seed)

    #### Start training process ####

    states = []
    actions = []
    reward_time = []

    folder_path = './training/'

    path_save = folder_path+ "testing_05/"

    ## Set up networks ##

    tf.reset_default_graph()

    mainQN = q_learning_pomdp.qnetwork(input_dim_v, output_dim, hidden_units, layers, learning_rate, clip_value,kernel_size,stride)
    targetQN = q_learning_pomdp.qnetwork(input_dim_v, output_dim, hidden_units, layers, learning_rate, clip_value,kernel_size,stride)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=None)

    trainables = tf.trainable_variables()

    targetOps = q_learning_pomdp.updateNetwork(trainables, tau)

    load_model = False
    ## Create replay buffer ##
    exp_buffer = q_learning_pomdp.replay_buffer(buffer_size)

    ## Randomness of actions ##
    epsilon = eStart
    stepDrop = (eStart - eEnd) / estep

    ## Further Variables
    done =False
    total_steps = 0

    with tf.Session() as sess:
        sess.run(init)
        start = time.time()
        for episode in range(max_train_episodes):
            episode_buffer = q_learning_pomdp.replay_buffer(buffer_size)
            env = world_pomdp.World(num_of_cars, num_of_lanes, track_length, speed_limit, ego_pos_init, ego_lane_init,
                              ego_speed_init,
                              dt, r_seed, x_range)

            observation = env.field_of_view()
            observation_v = processState(observation)

            reward_sum = 0
            timestep = 0
            action = 0
            done = False

            while done == False:
                if total_steps % 5 == 0:
                    if (np.random.random() < epsilon or total_steps < pre_train_steps):
                        action = random.randint(0,num_of_lanes*x_range-1)
                    else:
                        action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_scalar:[observation_v]})

                state1, reward,done = env.step(action)
                observation_1 = env.field_of_view()
                observation_1_v = processState(observation_1)



                total_steps += 1

                #episode_buffer.add(np.reshape(np.array([state_v,action,reward,state1_v,done]),[1,5]))
                episode_buffer.add(np.reshape(np.array([observation_v, action, reward, observation_1_v, done]), [1, 5]))
                if total_steps > pre_train_steps:
                    if epsilon > eEnd:
                        epsilon-=stepDrop

                    trainBatch = exp_buffer.sample(batch_size)
                    ## Calculate Q-Value: Q = r(s,a) + gamma*Q(s1,a_max)
                    # Use of the main network to predict the action a_max
                    action_max = sess.run(mainQN.action_pred,feed_dict={mainQN.input_scalar:np.vstack(trainBatch[:,3])})
                    Qt1_vec = sess.run(targetQN.output_q_predict,feed_dict={targetQN.input_scalar: np.vstack(trainBatch[:,3])}) #Q-values for s1

                    end_multiplier = -(trainBatch[:,4]-1)
                    Qt1 = Qt1_vec[range(batch_size),action_max] # select Q(s1,a_max)

                    # Q = r(s,a) + gamma*Q(s1,a_max)
                    Q_gt = trainBatch[:,2] + gamma*Qt1*end_multiplier

                    #Optimize network
                    _ = sess.run(mainQN.update,feed_dict={mainQN.input_scalar:np.vstack(trainBatch[:,0]),mainQN.q_gt:Q_gt,mainQN.actions:trainBatch[:,1]})

                    ## Update target network ##
                    if total_steps % update_freq == 0:
                        print("Update target network!")
                        q_learning_pomdp.updateTarget(targetOps,sess)

                reward_sum+=reward

                observation_v = observation_1_v
                #env.render()
            exp_buffer.add(episode_buffer.buffer)
            reward_sum_list[r_seed, episode] = reward_sum
            end = time.time()
            #print("Time for one episode: ",end-start," Episode: ",episode)
            if env.success == True:
                finished += 1

            if episode % 400 == 0:
                save_path = saver.save(sess,path_save+"modelRL_"+str(r_seed)+"_"+str(episode)+".ckpt")
                print("Model saved in: ",save_path)
            if episode % average_window == 0:
                if episode > 1:
                    print("Total steps: ", total_steps, " Episode: ", episode,
                          " Average reward over " + str(average_window) + " Episodes: ",
                          np.mean(reward_sum_list[r_seed, episode - average_window:episode]), "Finished: ", finished,
                          "/" + str(average_window) + "", " Episode:", episode, " Epsilon: ", epsilon)
                    #### Write to file ####
                    file = open(path_save + 'training_process' + str(r_seed) + '.txt', 'a')
                    file.write("Total steps: "+str(total_steps)+" Episode: "+ str(episode)+" Average reward over "+str(average_window)+" Episodes: "+
                          str(np.mean(reward_sum_list[r_seed,episode-average_window:episode]))+"Finished: "+str(finished)+"/"+str(average_window)+" Epsilon: "+str(epsilon)+"\n")

                    file.close()
                    #### Close File ####

                    reward_average[r_seed, int(episode / average_window)] = np.mean(reward_sum_list[r_seed,episode-average_window:episode])
                    finished_average[r_seed, int(episode / average_window)] = finished
                    finished = 0

            if r_seed == 0 and episode == 1:  # Only write for first time

                file = open(path_save + 'params_' + str(param_id) + '.txt', 'w')
                # file = open(complete_file, 'w')
                file.write('NETWORK PARAMETERS: \n\n')
                file.write('Layers: ' + str(layers) + '\n')
                file.write('Hidden units: ' + str(hidden_units) + '\n')
                file.write('Learning rate: ' + str(learning_rate) + '\n')
                file.write('Buffer size: ' + str(buffer_size) + '\n')
                file.write('Pre_train_steps: ' + str(pre_train_steps) + '\n')
                file.write('Batch_size: ' + str(batch_size) + '\n')
                file.write('Update frequency: ' + str(update_freq) + '\n')
                file.write('Tau: ' + str(tau) + '\n\n')

                file.write('RL PARAMETERS: \n\n')
                file.write('Gamma: ' + str(gamma) + '\n')
                file.write('Epsilon start: ' + str(eStart) + '\n')
                file.write('Epsilon end: ' + str(eEnd) + '\n')
                file.write('Epsilon steps: ' + str(estep) + '\n\n')

                file.write('SCENARIO PARAMETERS: \n\n')
                file.write('Episodes: ' + str(max_train_episodes) + '\n')
                file.write('Random sweeps: ' + str(random_sweep) + '\n')
                file.write('Cars: ' + str(num_of_cars) + '\n')
                file.write('Lanes: ' + str(num_of_lanes) + '\n')
                file.write('Ego speed init: ' + str(ego_speed_init) + '\n')
                file.write('Ego pos init: ' + str(ego_pos_init) + '\n')
                file.write('Ego lane init: ' + str(ego_lane_init) + '\n')
                file.write('Non-Ego tracklength: ' + str(track_length) + "\n\n\n")

                file.write('REMARKS: Big Network')

                file.close()


        final_save_path = saver.save(sess,path_save+"random_"+str(r_seed)+"_"  + "Final.ckpt")
        print("Model saved in: %s",final_save_path)

        end = time.time()
        print("Elapsed time for one cycle: ", end-start)
        file = open(path_save + 'training_process' + str(r_seed) + '.txt', 'a')
        file.write("Elapsed Time for one random seed:" + str(end-start)+"\n")




plt.figure(4)
ax = plt.subplot(1,1,1)
ax.set_title("Reward over time")
ax.set_xlabel("epsisode/100")
ax.set_ylabel("reward")
ax.grid()
sns.tsplot(reward_average)
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
plt.tight_layout()
plt.savefig(path_save + 'reward' + '.png')
plt.close()


plt.figure(5)
ax = plt.subplot(1, 1, 1)
ax.set_title("Success time")
ax.set_xlabel("epsiode/100")
ax.set_ylabel("Finished Episodes/100")
ax.grid()
sns.tsplot(finished_average)
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())
# plt.show(block=False)
plt.tight_layout()
plt.savefig(path_save + 'finished' + '.png')
# plt.show()
plt.close()


