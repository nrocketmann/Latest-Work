import numpy as np
from drone import Drone
from scipy.linalg import orth
#sets up the drone in a random location, orientation, velocity, and rotation
def randomize_drone(sim_agent, dist_max=5,vel_max=5,ang_max=6):
    start_location = np.random.random(size=[3])*dist_max
    start_velocity = np.random.random(size=[3])*vel_max
    start_angular = np.random.random(size=[3])*ang_max
    #graham schmidt to get orientation, assume full rank... hopefully that works
    start_orientation = orth(np.random.random(size=[3,3]))

    sim_agent.setOrientation(start_orientation[0],start_orientation[1])
    sim_agent.setAngular(start_angular)
    sim_agent.setPosition(start_location)
    sim_agent.setVelocity(start_velocity)

#for testing start with a stationary drone
def stationary_drone(sim_agent):
    sim_agent.setVelocity(np.zeros(shape=[3]))
    sim_agent.setAngular(np.zeros(shape=[3]))
    sim_agent.setPosition(np.zeros(shape=[3]))
    sim_agent.setOrientation(np.array([1,0,0]),np.array([0,1,0]))