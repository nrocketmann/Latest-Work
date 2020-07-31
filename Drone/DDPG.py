print("importing")
from drone import Drone
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from scipy.linalg import orth
import time
import random
from drone_utils import *
print("done, now running the actual goddamn program")

use_existing = False

#define the drone, we'll update its position before each simulation
print("building drone")
drone_mass = 1
max_rotor_force = 5
rotor_torque = 0
arm_length = .2
agent = Drone(drone_mass,max_rotor_force,rotor_torque,arm_length)

#distances along each axis in meters
max_dist_axis = 1
max_vel_axis = 2
max_angular_axis = np.pi

#training constants
dt = .001
sim_updates = 5 #how many simulator updates per one python update
sim_time = 2 #total time simulated
iterations = int(np.floor(sim_time/(dt*sim_updates)))
print('iterations of python code for one sim: ' + str(iterations))
print('total updates of drone for one sim: ' + str(iterations*(sim_updates)))
print('ms model is running at: ' + str(sim_time/iterations*1000))
print('model fps: ' + str(iterations/sim_time))
print('simulator fps: ' + str(sim_updates*iterations/sim_time))

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.states_buffer = []
        self.next_states_buffer = []
        self.rewards_buffer = []
        self.actions_buffer = []

    def add_run(self, states, next_states, actions, rewards):
        size = len(states)
        if (size+len(self.states_buffer))>self.max_size:
            self.reset()
        self.states_buffer+=states
        self.next_states_buffer+=next_states
        self.actions_buffer+=actions
        self.rewards_buffer+=rewards

    def fetch(self,fetch_size):
        inds = np.random.choice(len(self.states_buffer),fetch_size)
        states_chunk = [self.states_buffer[ind] for ind in inds]
        next_states_chunk = [self.next_states_buffer[ind] for ind in inds]
        rewards_chunk = [self.rewards_buffer[ind] for ind in inds]
        actions_chunk = [self.actions_buffer[ind] for ind in inds]
        return np.array(states_chunk).astype(np.float32).reshape([fetch_size, 18]), \
               np.array(next_states_chunk).astype(np.float32).reshape([fetch_size, 18]), \
               np.array(actions_chunk).astype(np.float32).reshape([fetch_size, 4]),\
               np.array(rewards_chunk).astype(np.float32).reshape([fetch_size])

    def reset(self):
        self.shuffle()
        half_size = self.max_size//2
        self.states_buffer = self.states_buffer[:half_size]
        self.next_states_buffer = self.next_states_buffer[:half_size]
        self.rewards_buffer = self.rewards_buffer[:half_size]
        self.actions_buffer = self.actions_buffer[:half_size]

    def shuffle(self):
        together = list(zip(self.states_buffer, self.next_states_buffer, self.rewards_buffer, self.actions_buffer))
        random.shuffle(together)
        self.states_buffer, self.next_states_buffer, self.rewards_buffer, self.actions_buffer = zip(*together)
        self.states_buffer = list(self.states_buffer)
        self.next_states_buffer = list(self.next_states_buffer)
        self.rewards_buffer = list(self.rewards_buffer)
        self.actions_buffer = list(self.actions_buffer)

class Actor(Model):
    def __init__(self, layer_sizes):
        super(Actor,self).__init__()
        fc_layers = []
        for layer_size in layer_sizes:
            fc_layers.append(Dense(layer_size,activation='tanh'))
        self.fc_layers = fc_layers
        self.output_layer = Dense(4,activation='sigmoid')

    def call(self,inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)
        return self.output_layer(h)


class Critic(Model):
    def __init__(self, layer_sizes):
        super(Critic,self).__init__()
        fc_layers = []
        for layer_size in layer_sizes:
            fc_layers.append(Dense(layer_size,activation='tanh'))
        self.fc_layers = fc_layers
        self.output_layer = Dense(1,activation=None)


    def call(self,inputs):
        state, action = inputs
        h = tf.concat([state,action],axis=-1) #22 dims
        for layer in self.fc_layers:
            h = layer(h)
        return self.output_layer(h)

#constants relating to the models
actor_layers = [64,32,16]
critic_layers = [64,32,16]
critic_lrate = 1e-6
actor_lrate = 1e-3
polyak = 0.98
update_batch = 64
gamma = tf.constant(.98)
buffer_max_size = 10000
updates_per_sim = 1


#create networks
if not use_existing:
    actor = Actor(actor_layers)
    critic = Critic(critic_layers)
    actor_target = Actor(actor_layers)
    critic_target = Critic(critic_layers)
else:
    actor = tf.keras.models.load_model('actor.pb')
    critic = tf.keras.models.load_model('critic.pb')
    actor_target = tf.keras.models.load_model('actor_target.pb')
    critic_target = tf.keras.models.load_model('critic_target.pb')



# actor_target.trainable = False
# critic_target.trainable = False
optimizer_critic = tf.keras.optimizers.SGD(critic_lrate)
optimizer_actor = tf.keras.optimizers.SGD(actor_lrate)
replay_buffer = ReplayBuffer(buffer_max_size)

#functions to update the target networks
@tf.function
def update_actor_target(target_polyak):
    # actor_target.trainable = True
    for network_var, target_var in zip(actor.trainable_variables, actor_target.trainable_variables):
        target_var.assign(target_polyak*target_var + (1-target_polyak)*network_var)
    # actor_target.trainable = False

@tf.function
def update_critic_target(target_polyak):
    for network_var, target_var in zip(critic.trainable_variables, critic_target.trainable_variables):
        target_var.assign(target_polyak*target_var + (1-target_polyak)*network_var)

#function to update main networks and target networks all at once
@tf.function
def update(states, next_states, actions, rewards, target_polyak):
    critic_loss_returnable = 0
    actor_expected_reward = 0
    actor_target.trainable = False
    critic_target.trainable = False

    with tf.GradientTape() as tape:
        #update the critic first
        next_actions = actor_target(next_states)
        tf.debugging.check_numerics(next_actions,"target actions bad")
        target_val = rewards + gamma * critic_target([next_states,next_actions])
        tf.debugging.check_numerics(target_val,"critic target bad")
        critic_loss = tf.square(critic([states,actions])-target_val) * 1/update_batch
        tf.debugging.check_numerics(critic_loss, "critic loss bad")
        critic_loss_returnable+=tf.reduce_sum(critic_loss)
        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        #tf.print(tf.reduce_sum([tf.reduce_sum(tf.math.abs(grad))  for grad in critic_grad]))
        optimizer_critic.apply_gradients(zip(critic_grad,critic.trainable_variables))
    critic.trainable = False
    with tf.GradientTape() as tape:
        #now update the actor
        actor_loss = -critic([states, actor(states)]) * 1/update_batch
        tf.debugging.check_numerics(actor_loss, "actor loss bad")
        actor_expected_reward+=tf.reduce_sum(-actor_loss)
        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
        #tf.print(tf.reduce_sum([tf.reduce_sum(tf.math.abs(grad)) for grad in actor_grad]))
        optimizer_actor.apply_gradients(zip(actor_grad,actor.trainable_variables))

    critic.trainable = True
    actor_target.trainable = True
    critic_target.trainable = True
    update_actor_target(target_polyak)
    update_critic_target(target_polyak)
    return critic_loss_returnable, actor_expected_reward

def alternative_reward(state, bubble_size):
    reward = 0
    state = state.squeeze()
    position = state[:3]
    velocity = state[3:6]
    angular = state[6:9]
    orientation = state[9:]
    if (np.linalg.norm(position)<bubble_size):
        reward+=1
    if np.linalg.norm(velocity)<bubble_size:
        reward+=1
    if np.linalg.norm(angular)<bubble_size:
        reward+=1
    if np.linalg.norm(orientation-np.identity(3).flatten())<bubble_size:
        reward+=1
    return reward

def add_noise(actions, stdev):
    noise = np.random.normal(0,stdev,[4,1])
    return np.clip(actions + noise,0,1)

#training constants
epochs = 20000

def run_epoch(noise_stdev=.1):

    #setup
    #randomize_drone(max_dist_axis, max_vel_axis, max_angular_axis, agent)
    stationary_drone(agent)
    states = []
    next_states = []
    actions = []
    rewards = []
    states.append(agent.getVector().reshape([1,18]))

    #simulator run
    for i in range(iterations):
        action = actor(states[-1]).numpy().reshape([4,1])
        action = add_noise(action, noise_stdev*i/iterations)
        actions.append(action)
        next_state = agent.fullIteration(action, dt, sim_updates).reshape([1,18])
        #next_state = agent.fullIteration(np.array([1.0,1.0,1.0,1.0]), dt, sim_updates).reshape([1, 18])
        next_states.append(next_state)
        reward = -agent.getLoss()
        #reward = alternative_reward(next_state, .5)
        rewards.append(reward)
        if (reward==0 and i>update_batch and i!=iterations-1):
            break
        else:
            if (i!=iterations-1):
                states.append(next_state)
    #add to buffer
    # rewards = (-np.array(rewards)/(np.min(rewards)-1)).tolist()
    rewards = (-np.log(-np.array(rewards)+1)).tolist()
    #print(rewards)
    reward_total = np.sum(rewards)
    replay_buffer.add_run(states,next_states,actions,rewards)

    for _ in range(updates_per_sim):
        #fetch from buffer and update
        states_chunk, next_states_chunk, actions_chunk, rewards_chunk = replay_buffer.fetch(update_batch)
        printable_critic_loss, printable_actor_expectation = update(states_chunk, next_states_chunk, actions_chunk, rewards_chunk, polyak)


    return reward_total, printable_critic_loss, printable_actor_expectation

#build the models by calling once
run_epoch(0.0)

#look at the models rawwrr
actor.summary()
critic.summary()

#printables
print_number = 50
total_reward = 0
total_critic_loss = 0
total_actor_expectation = 0

#actual main train loop
for i in range(epochs):

    #on first time through make networks identical
    if i==0 and use_existing==False:
        update_actor_target(0.0)
        update_critic_target(0.0)

    #the actual epoch
    reward_val, critic_loss_val,actor_expectation_val  = run_epoch(0.1)

    #printables
    total_reward+=reward_val
    total_critic_loss+=critic_loss_val
    total_actor_expectation+=actor_expectation_val
    if (i+1)%print_number==0:
        print("Average reward at epoch " + str(i+1) + ": " + str(total_reward/print_number))
        print("Average critic loss at epoch " + str(i + 1) + ": " + str(critic_loss_val / print_number))
        print("Average actor expectation at epoch " + str(i + 1) + ": " + str(total_actor_expectation / print_number))
        total_reward = 0
        total_loss = 0
        total_actor_expectation = 0
        print(np.mean(np.array(replay_buffer.actions_buffer),axis=0))

        actor.save('actor')
        critic.save('critic')
        actor_target.save('actor_target')
        critic_target.save('critic_target')