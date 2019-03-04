import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

#### Import own modules ####
import sys
sys.path.insert(0,'environment/')
import q_learning
import world
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


def relative_state(state):

    for id_n in range(len(state)-1,-1,-1):
        state[id_n].x = state[id_n].x-state[0].x
        state[id_n].y = state[id_n].y - state[0].y
        state[id_n].v = state[id_n].v - state[0].v

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
clip_value = 7500
learning_rate = 0.001
buffer_size = 100000
batch_size = 32
update_freq = 20000

#### RL Parameters ####

gamma = 0.99
eStart = 1
eEnd = 0.1
estep = 150000

#### Learning Parameters ####

max_train_episodes = 150000
pre_train_steps = 100000
random_sweep = 3
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

#### Plot variables ####
max_timestep = 400
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
enable = 10

for r_seed in range(0,random_sweep):

    random.seed(r_seed)

    #### Start training process ####

    states = []
    actions = []
    reward_time = []

    folder_path = './training/'

    path_save = folder_path+ "results_03/"

    ## Set up networks ##

    tf.reset_default_graph()

    mainQN = q_learning.qnetwork(input_dim, output_dim, hidden_units, layers, learning_rate, clip_value)
    targetQN = q_learning.qnetwork(input_dim, output_dim, hidden_units, layers, learning_rate, clip_value)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=None)

    trainables = tf.trainable_variables()

    targetOps = q_learning.updateNetwork(trainables, tau)

    load_model = False
    ## Create replay buffer ##
    exp_buffer = q_learning.replay_buffer(buffer_size)

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
            episode_buffer = q_learning.replay_buffer(buffer_size)

            #if episode % 200 == 0:
            #    ego_pos_init = 100 * random.random()
            #    ego_lane_init = random.randint(1, 2)

            env = world.World(num_of_cars, num_of_lanes, track_length, speed_limit, ego_pos_init, ego_lane_init,
                              ego_speed_init,
                              dt, r_seed, x_range)
            state,_,_ = env.get_state()
            state = relative_state(state)
            state_v = vectorize_state(state)


            reward_sum = 0
            timestep = 0
            action = 0
            done = False

            while done == False:
                if total_steps % enable == 0:
                    if (np.random.random() < epsilon or total_steps < pre_train_steps):
                        action = random.randint(0,num_of_lanes*x_range-1)
                        #print("RANDOM)")
                    else:
                        action = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:[state_v]})
                        action = action[0]
                        #print("Network")

                state1, reward,done = env.step(action)
                state1 = relative_state(state1)
                state1_v = vectorize_state(state1)

                total_steps += 1

                episode_buffer.add(np.reshape(np.array([state_v,action,reward,state1_v,done]),[1,5]))

                if total_steps > pre_train_steps:
                    if epsilon > eEnd:
                        epsilon-=stepDrop

                    trainBatch = exp_buffer.sample(batch_size)
                    ## Calculate Q-Value: Q = r(s,a) + gamma*Q(s1,a_max)
                    # Use of the main network to predict the action a_max
                    action_max = sess.run(mainQN.action_pred,feed_dict={mainQN.input_state:np.vstack(trainBatch[:,3])})
                    Qt1_vec = sess.run(targetQN.output_q_predict,feed_dict={targetQN.input_state: np.vstack(trainBatch[:,3])}) #Q-values for s1

                    end_multiplier = -(trainBatch[:,4]-1)
                    Qt1 = Qt1_vec[range(batch_size),action_max] # select Q(s1,a_max)

                    # Q = r(s,a) + gamma*Q(s1,a_max)
                    Q_gt = trainBatch[:,2] + gamma*Qt1*end_multiplier

                    #Optimize network
                    _ = sess.run(mainQN.update,feed_dict={mainQN.input_state:np.vstack(trainBatch[:,0]),mainQN.q_gt:Q_gt,mainQN.actions:trainBatch[:,1]})

                    ## Update target network ##
                    if total_steps % update_freq == 0:
                        print("Update target network!")
                        q_learning.updateTarget(targetOps,sess)

                reward_sum+=reward

                state_v = state1_v
                #env.render()
            exp_buffer.add(episode_buffer.buffer)
            reward_sum_list[r_seed, episode] = reward_sum
            end = time.time()
            #print("Time for one episode: ",end-start," Episode: ",episode)
            if env.success == True:
                finished += 1

            if episode % 10000 == 0:
                save_path = saver.save(sess,path_save+"modelRL_"+str(r_seed)+"_"+str(episode)+".ckpt")
                print("Model saved in: ",save_path)
            if episode % average_window == 0:
                if episode > 1:
                    print("Total steps: ", total_steps," Episode: ",episode, " Average reward over "+str(average_window)+" Episodes: ",
                          np.mean(reward_sum_list[r_seed,episode-average_window:episode]),"Finished: ",finished,"/"+str(average_window)+""," Episode:", episode, " Epsilon: ",epsilon)
                    reward_average[r_seed, int(episode / average_window)] = np.mean(reward_sum_list[r_seed,episode-average_window:episode])
                    finished_average[r_seed, int(episode / average_window)] = finished
                    #### Write to file ####
                    file = open(path_save + 'training_process' + str(r_seed) + '.txt', 'a')
                    file.write("Total steps: " + str(total_steps) + " Episode: " + str(
                        episode) + " Average reward over " + str(average_window) + " Episodes: " +
                               str(np.mean(
                                   reward_sum_list[r_seed, episode - average_window:episode])) + "Finished: " + str(
                        finished) + "/" + str(average_window) + "\n")

                    file.close()
                    #### Close File ####


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
                file.write('Action buffer' + str(enable) + '\n')
                file.write('Ego speed init: ' + str(ego_speed_init) + '\n')
                file.write('Ego pos init: ' + str(ego_pos_init) + '\n')
                file.write('Ego lane init: ' + str(ego_lane_init) + '\n')
                file.write('Non-Ego tracklength: ' + str(track_length) + "\n\n\n")

                file.write('REMARKS: Reward 2, 5 degree Run \n\n\n\n')


                file.close()

        final_save_path = saver.save(sess,path_save+"modelRL_"+str(r_seed)+"_"  + "final.ckpt")
        print("Model saved in: %s",final_save_path)




plt.figure(4)
ax = plt.subplot(1,1,1)
ax.set_title("Reward over time")
ax.set_xlabel("episode/100")
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
ax.set_xlabel("episode/100")
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


