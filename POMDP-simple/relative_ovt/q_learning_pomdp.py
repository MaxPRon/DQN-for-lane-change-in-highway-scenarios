import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random





class qnetwork:

    def __init__(self,input_dim,output_dim,hidden_units,layers,learning_rate,clip_value,kernel_size=[4,4],stride=[2,2]):

        # Input
        self.input_scalar = tf.placeholder(shape=[None,6622],dtype=tf.float32,name="input_placeholder")
        #self.input_state = tf.placeholder(tf.float32,input_dim,name = "input_placeholder")
        self.input_state = tf.reshape(self.input_scalar,shape=[-1,301,11,2])
        #self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        # Network Architecture
        #self.hidden_layer = tf.layers.dense(self.input_state,hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.hidden_layer = slim.conv2d(inputs=self.input_state,num_outputs=hidden_units,kernel_size=kernel_size,stride=stride, padding='VALID',biases_initializer=None)
        for n in range(1,layers):
            #self.hidden_layer = tf.layers.dense(self.hidden_layer,hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.hidden_layer = slim.conv2d(inputs=self.hidden_layer, num_outputs=hidden_units, kernel_size=kernel_size,
                                            stride=stride, padding='VALID', biases_initializer=None)

        #Calculate Neccessar

        # Final Convlayer #### Maybe change Structure ####
        self.hidden_layer = slim.conv2d(inputs=self.hidden_layer, num_outputs=hidden_units, kernel_size=[self.hidden_layer.shape[1],self.hidden_layer.shape[2]],
                                       stride=[1,1], padding='VALID', biases_initializer=None)

        self.hidden_layer_flat = tf.layers.flatten(self.hidden_layer)
        #### Implementation Dueling DQN
        ## Q(s,a) = V(s) + A(s,a)

        ## Calculation V(s)
        self.value_fc = tf.layers.dense(inputs=self.hidden_layer_flat,units=hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.value = tf.layers.dense(inputs=self.value_fc,units=1,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

        ## Calculation A(s,a)
        self.advantage_fc = tf.layers.dense(inputs = self.hidden_layer_flat,units=hidden_units,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.advantage = tf.layers.dense(inputs=self.advantage_fc,units=output_dim,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.output_q_predict = self.value + tf.subtract(self.advantage,tf.reduce_mean(self.advantage,axis=1,keepdims=True))

        #self.output_q_predict = tf.layers.dense(self.hidden_layer,output_dim)
        # Clip values just in case
        self.output_q_predict = tf.clip_by_value(self.output_q_predict,-clip_value,clip_value)
        # Get action (highest q-value)
        self.action_pred = tf.argmax(self.output_q_predict,1) # second axis

        # Compute Cost/Loss
        self.actions = tf.placeholder(tf.int32,shape = [None])
        self.q_gt = tf.placeholder(tf.float32, [None]) # Q-value groundtruth
        # Encode into onehot to select q-value
        self.actions_onehot = tf.one_hot(self.actions,output_dim)

        # select single Q-value given the action
        self.q_action = tf.reduce_sum(tf.multiply(self.output_q_predict,self.actions_onehot),axis = 1)
        self.absolute_error = abs(self.q_gt - self.q_action)
        self.cost = tf.losses.mean_squared_error(self.q_gt,self.q_action)

        #self.per_cost = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.q_gt,self.q_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer_per = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.minimize(self.cost)
        #self.update_per = self.optimizer_per.minimize(self.per_cost)


#### Design Replay buffer

class replay_buffer():
    def __init__(self,buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size


    def add(self,exp):
        #### Check if buffer full
        if(len(self.buffer)+ len(exp) >= self.buffer_size):
            # Remove oldest exp which is too much
            self.buffer[0:(len(exp)+ len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(exp)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5]) # state,action, reward,state_1, done


#### Helper function for target network update

def updateNetwork(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []

    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value()*tau)+ ((1-tau)*tfVars[idx+total_vars//2].value())))
    return  op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def bayes_objective(reward,window):

    objective = np.mean(reward[-window:],axis=0)
    objective = np.mean(objective)

    return objective


class SumTree(object):

    data_pointer = 0

    def __init__(self,capacity):

        self.capacity = capacity

        self.tree = np.zeros(2*capacity-1) #tree indices

        self.experience_data = np.zeros(capacity,dtype=object)


    def add(self,priority,data):
        # Look at what index to put the experience
        tree_index = self.data_pointer + self.capacity-1

        # Update data frame
        self.experience_data[self.data_pointer] = data

        self.update(tree_index,priority)

        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0


    def update(self,tree_index,priority):

        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1)//2
            self.tree[tree_index]+= change

    def get_leaf(self,v):

        parent_index = 0

        while True:
            left_child_index = 2*parent_index+1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:

                if(v <= self.tree[left_child_index]):
                    parent_index = left_child_index

                else:
                    v-=self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity+1

        return leaf_index, self.tree[leaf_index], self.experience_data[data_index]

    def total_priority(self):
        return self.tree[0]


class prioritized_experience_buffer(object):

    PER_e = 0.01 # Constant for probability of selection never 0

    PER_a = 0.6 # Tradeoff between random and prio sampling

    PER_b = 0.4 # importance sampling, increas to 1

    PER_b_steps = 50000


    PER_b_step_size = (1-PER_b)/PER_b_steps

    absolute_error_upper = 1

    def __init__(self,capacity):

        self.tree = SumTree(capacity)

    def store(self, experience):

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority,experience)


    def sample(self, n):

        memory_b = []

        b_idx, b_ISWeights = np.empty((n,),dtype=np.int32), np.empty((n,1),dtype=np.float32)

        priority_segment = np.divide(self.tree.total_priority(), n)


        self.PER_b = np.min([1.,self.PER_b + self.PER_b_step_size])

        p_min = np.amin(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority()
        if p_min == 0:
            p_min = 0.00001
        max_weight = (p_min*n)**(-self.PER_b)

        for i in range(n):

            a, b = priority_segment * i, priority_segment*(i+1)
            value = np.random.uniform(a,b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority()

            b_ISWeights[i,0] = np.power(n*sampling_probabilities,-self.PER_b)/ max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    def batch_update(self,tree_idx,abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors,self.absolute_error_upper)
        ps = np.power(clipped_errors,self.PER_a)

        for ti, p in zip(tree_idx,ps):
            self.tree.update(ti,p)

