# Quadcopter Attitude Control with Reinforcement Learning
For years, I've thought it would be amazing to be able to fly a drone entirely with machine learning. When I first began set out to put my idea into practice 
over two years ago, I very quickly ran into a wall: I needed a quality physics simulator to generate training data. Most of the popular simulators I found were massive
packages, far exceeding my needs and far too slow to generate the data I required unless I put substantial cash into renting an AWS or GCP instance. After
setting the project aside for close to two years, I came across [this paper](https://arxiv.org/pdf/1707.05110.pdf), in which the authors built their own simulator from scratch.
After some brief research on the equations for motion of a rigid body, I quickly put together a simulator in C++ with python bindings, enabling me to easily build
Reinforcement Learning algorithms in python capable of interacting with the environment quickly and efficiently. Afterward, I built implementations of
[Deep Deterministic Policy Graident (DDPG)](https://arxiv.org/pdf/1509.02971.pdf), [Proximal Policy Optimization](https://arxiv.org/pdf/1509.02971.pdf), and
a third algorithm of my own invention that involves training a differentiable network to model the environment. In the future, I hope to put these models on a physical
drone and see if the device is capable of flying despite the simplifications in the simulator.
