from drone import Drone
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from scipy.linalg import orth
import time
import random
from drone_utils import *
import tensorflow_probability as tfp

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
sim_updates = 1 #how many simulator updates per one python update
sim_time = 2 #total time simulated
iterations = int(np.floor(sim_time/(dt*sim_updates)))
print('iterations of python code for one sim: ' + str(iterations))
print('total updates of drone for one sim: ' + str(iterations*(sim_updates)))
print('ms model is running at: ' + str(sim_time/iterations*1000))
print('model fps: ' + str(iterations/sim_time))
print('simulator fps: ' + str(sim_updates*iterations/sim_time))

class Actor(Model):
    def __init__(self, layer_sizes):
        super(Actor,self).__init__()
        fc_layers = []
        for i, layer_size in enumerate(layer_sizes):
            if i==len(layer_sizes)-1:
                fc_layers.append(Dense(layer_size,activation=None))
            else:
                fc_layers.append(Dense(layer_size,activation='tanh'))
        self.fc_layers = fc_layers
        self.mean_layer = Dense(4,activation='sigmoid')
        self.stdev_layer = Dense(4,activation='sigmoid')

    def call(self,inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)

        means,stdevs = self.mean_layer(h), self.stdev_layer(h)
        return means,stdevs


class Critic(Model):
    def __init__(self, layer_sizes):
        super(Critic,self).__init__()
        fc_layers = []
        for layer_size in layer_sizes:
            fc_layers.append(Dense(layer_size,activation='tanh'))
        self.fc_layers = fc_layers
        self.output_layer = Dense(1,activation=None)


    def call(self,inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)
        out = self.output_layer(h)
        return out

policy = Actor([64,128,256])
value = Critic([128,128,128])
old_policy = Actor([64,128,256])

policy_lrate = 1e-5
value_lrate = 1e-4

policy_optimizer = tf.keras.optimizers.Adam(policy_lrate)
value_optimizer = tf.keras.optimizers.Adam(value_lrate)

value.compile(loss='mse',optimizer=value_optimizer)

gamma = .98
epsilon = .2
reset_old_policy_iter = 2

#just sets the old policy model to be the same as the new one
@tf.function
def set_old_policy():
    for old_var, new_var in zip(old_policy.trainable_variables, policy.trainable_variables):
        old_var.assign(new_var)

#takes an action chosen stochastically from the distribution outputted by the policy
@tf.function
def make_action(inputs):
    means,stdevs = policy(inputs)
    tf.debugging.check_numerics(means, "floof")
    tf.debugging.check_numerics(stdevs,"fouff")
    distro = tfp.distributions.Normal(means,stdevs).sample(1)
    return tf.clip_by_value(distro,0.0,1.0)

@tf.function(experimental_relax_shapes=True)
def update_policy(states, advantages, actions):
    with tf.GradientTape() as tape:
        means, stds = policy(states)
        old_means, old_stds = old_policy(states)
        action_probs = tfp.distributions.Normal(means,stds).prob(actions)
        old_action_probs = tfp.distributions.Normal(old_means, old_stds).prob(actions)
        rt = tf.multiply(tf.multiply(action_probs, tf.math.reciprocal(old_action_probs)), advantages)
        clipper = tf.multiply(tf.clip_by_value(rt, 1-epsilon, 1+epsilon), advantages)
        L = tf.math.minimum(rt,clipper) #maximize this
        L_min = -L #minimize this

        grads = tape.gradient(L_min,policy.trainable_variables)
        #grad_sum = tf.reduce_sum([tf.reduce_sum(g) for g in grads])
        #tf.print(grad_sum)
        policy_optimizer.apply_gradients(zip(grads, policy.trainable_variables))

def get_targets(rewards, last_state):
    targets = []
    last_value = value(last_state.reshape([1,18])).numpy().squeeze()
    targ = last_value
    for reward in reversed(rewards):
        targ = reward + gamma*targ
        targets.append(targ)
    return np.array(targets)

def run_epoch():
    stationary_drone(agent)
    rewards = [] #should all be negative
    states = []
    actions = []

    # pos_loss_sum = 0
    # or_loss_sum = 0
    # vel_loss_sum = 0
    # ang_loss_sum = 0

    state = agent.getVector().reshape([1,18])

    for i in range(iterations):
        r = -agent.getLoss()
        rewards.append(r)

        #for debugging
        # pos_loss_sum+=np.linalg.norm(agent.getPosition())
        # or_loss_sum+=np.linalg.norm(agent.getOrientation().flatten()-np.identity(3).flatten())
        # vel_loss_sum+=np.linalg.norm(agent.getVelocity())
        # ang_loss_sum+=np.linalg.norm(agent.getAngular())

        states.append(state)
        action = make_action(state).numpy().reshape([4,1])
        actions.append(action)

        if r<-100 or np.linalg.norm(agent.getAngular())>100:
            break
        state = agent.fullIteration(action, dt, sim_updates).reshape([1,18])

    #do all the updates
    states = np.array(states)
    last_state = state
    #targets for value net
    targets = get_targets(rewards, last_state)

    advantages = value(np.array(states))

    value_loss = value.train_on_batch(states,targets) #update value net
    update_policy(np.array(states) ,np.array(advantages),np.array(actions)) #update policy
    #print(rewards)
    return np.sum(rewards), value_loss, len(rewards)#, pos_loss_sum, vel_loss_sum, or_loss_sum, ang_loss_sum

epochs = 25000
print_num = 1000
reward_sum = 0
value_losses = 0
reward_lengths = 0

for i in range(epochs):
    if i%reset_old_policy_iter==0:
        set_old_policy()

    # r, l, rl, pls, vls, ols, als = run_epoch()
    r, l, rl = run_epoch()
    #print("pos loss: " + str(pls))
    #print("vel loss: " + str(vls))
    #print("or loss: " + str(ols))
    #print("ang loss: " + str(als))


    reward_sum+=r
    value_losses+=l
    reward_lengths+=rl

    if (i+1)%print_num==0:
        print("Average reward at iteration " + str(i+1) + ": " + str(reward_sum/print_num))
        print("Average value loss at iteration " + str(i+1) + ": " + str(value_losses/print_num))
        print("Average reward length at iteration " + str(i + 1) + ": " + str(reward_lengths / print_num))
        reward_sum = 0
        value_losses = 0
        reward_lengths = 0

policy.save("ppo_policynet")
value.save("ppo_valuenet")

