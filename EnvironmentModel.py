from drone import Drone
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from scipy.linalg import orth
import time
import random
from drone_utils import *
import tensorflow_probability as tfp

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
        self.last_layer = Dense(4,activation='sigmoid')

    def call(self,inputs):
        h = inputs
        for layer in self.fc_layers:
            h = layer(h)
        thrusts = self.last_layer(h)
        return thrusts


class Environment(Model):
    def __init__(self, layer_sizes):
        super(Environment,self).__init__()
        fc_layers = []
        for layer_size in layer_sizes:
            fc_layers.append(Dense(layer_size,activation='tanh'))
        self.fc_layers = fc_layers
        self.output_layer = Dense(18,activation=None)


    def call(self,inputs):
        states, actions = inputs
        h = tf.concat([states,actions],axis=1)
        for layer in self.fc_layers:
            h = layer(h)
        return self.output_layer(h)

policy = Actor([64,128,256])
environment = Environment([128,128,128])

policy_lrate = 1e-5
environment_lrate = 1e-4

policy_optimizer = tf.keras.optimizers.Adam(policy_lrate)
environment_optimizer = tf.keras.optimizers.Adam(environment_lrate)

#environment compiled so it can train on simulator samples
environment.compile(loss='mse',optimizer=environment_optimizer)

#now we want a big model that takes in a state, predicts an action, and then calculates reward based on env model
environment.trainable = False
big_model_inp = Input([18])
big_model_action = policy(big_model_inp)
next_state_pred = environment([big_model_inp, big_model_action])
pos_loss = tf.linalg.norm(next_state_pred[:,:3])
vel_loss = tf.linalg.norm(next_state_pred[:,3:6])
ang_loss = tf.linalg.norm(next_state_pred[:,6:9])
good_orientation = tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
or_loss = tf.linalg.norm(next_state_pred[:,9:]-good_orientation)
big_model_loss = vel_loss + ang_loss + or_loss + pos_loss
big_model = Model(inputs=big_model_inp, outputs = big_model_loss)
big_model.compile(loss='mse', optimizer=policy_optimizer)

class EnvironmentTrainer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states_buffer = []
        self.next_buffer = []
        self.action_buffer = []

    def random_sample(self):
        randomize_drone(agent)
        for i in range(iterations):
            state = agent.getVector()
            self.states_buffer.append(state)
            action = np.random.random([4,1])
            assert 0<=np.min(action) and 1>=np.max(action)
            state = agent.fullIteration(action, dt, sim_updates).reshape([18])
            self.next_buffer.append(state)
            self.action_buffer.append(action.reshape([4]))

            if np.isnan(self.states_buffer[-1]).any() or np.isnan(self.action_buffer[-1]).any()\
                    or np.isnan(self.next_buffer[-1]).any() or np.max(np.abs(state))>100:
                self.states_buffer.pop()
                self.next_buffer.pop()
                self.action_buffer.pop()
                break

    def on_policy_sample(self,stationary=True):
        if stationary:
            stationary_drone(agent)
        else:
            randomize_drone(agent)
        for i in range(iterations):
            state = agent.getVector().reshape([1,18])
            self.states_buffer.append(state.reshape([18]))
            action = policy(state).numpy().reshape([4,1])
            assert 0<=np.min(action) and 1>=np.max(action)
            state = agent.fullIteration(action, dt, sim_updates).reshape([18])
            self.next_buffer.append(state)
            self.action_buffer.append(action.reshape([4]))

            if np.isnan(self.states_buffer[-1]).any() or np.isnan(self.action_buffer[-1]).any()\
                    or np.isnan(self.next_buffer[-1]).any() or np.max(np.abs(state))>100:
                self.states_buffer.pop()
                self.next_buffer.pop()
                self.action_buffer.pop()
                break

    def fit_batch(self, batch_size, epochs=1):
        print(np.max(np.array(self.states_buffer)))

        inds = np.random.choice(np.arange(len(self.states_buffer)),[batch_size], replace=False)
        states_chunk = np.array([self.states_buffer[i] for i in inds])
        actions_chunk = np.array([self.action_buffer[i] for i in inds])
        next_chunk = np.array([self.next_buffer[i] for i in inds])
        environment.fit([states_chunk, actions_chunk], next_chunk, epochs=epochs, verbose=1)

    def fit_environment_batch(self):
        l = environment.train_on_batch([np.array(self.states_buffer), np.array(self.action_buffer)], np.array(self.next_buffer))
        print("environment loss: " + str(l))

    def fit_policy(self):
        l = big_model.train_on_batch(np.array(self.states_buffer), np.zeros(len(self.states_buffer)))
        print("policy loss: " + str(l))

    def get_size(self):
        return len(self.states_buffer)

trainer = EnvironmentTrainer()
#pretrain the environment model
try:
    environment = tf.keras.models.load_model("pretrained_environment")
    environment.compile(loss='mse',optimizer=environment_optimizer)
    print("environment model loaded from file")
except KeyError:
    print("gathering data to train environment")
    for i in range(2000):
        trainer.random_sample()
    print("training environment")
    trainer.fit_batch(trainer.get_size(),epochs=15)
    environment.save('pretrained_environment')

for i in range(100000):
    trainer.on_policy_sample()
    if ((i+1)%10==0):
        #trainer.fit_environment_batch()
        trainer.fit_policy()
        trainer.clear()









