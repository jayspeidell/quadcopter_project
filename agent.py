import numpy as np
from task import Task
from actor_critic import make_actor, make_critic
from replay import PrioritizedReplay

# https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html


class Agent:
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None,
                 buffer_size=150000, batch_size=32, alpha=1, beta=1, gamma=0.1,
                 replay_alpha=0.9):

        self.task = Task(init_pose, init_velocities,
                         init_angle_velocities, runtime, target_pos)

        self.state_size = self.task.state_size
        self.action_size = self.task.action_size

        self.state = self.task.sim.reset()

        self.memory = PrioritizedReplay(buffer_size, batch_size, replay_alpha)

        self.actor = make_actor(self.state_size, self.action_size)
        self.actor_weights = self.actor.trainable_weights
        self.actor_target = make_actor(self.state_size, self.action_size)

        self.critic = make_critic(self.state_size, self.action_size)
        self.critic_weights = self.critic.trainable_weights
        self.critic_target = make_critic(self.state_size, self.action_size)

        self.gamma = 0.1

    def step(self):
        self.act()

        if self.memory.current_memory > self.memory.batch_size:
            self.learn()




    def act(self):
        action = self.actor(self.state)
        next_state, reward, done = self.task.step(action)
        Q = self.critic(self.state, action)
        Q_h = self.critic(next_state, action)
        TD = reward + self.gamma * Q_h - Q
        value = abs(TD)
        self.memory.add(self.state, action, reward, next_state, done, value)

    def learn(self):
        print('hi')

    def play(self):
        print('hi')

    def update_target(self, target):
        self.task.new_target(target)

    def save_model(self,name):
        print('hi')

    def load_model(self,name):
        print('hi')