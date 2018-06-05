import numpy as np
from task import Task
from actor_critic import Actor, Critic
from replay import PrioritizedReplay
import keras.backend as K
import tensorflow as tf


class Agent:
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None,
                 buffer_size=150000, batch_size=32, alpha=1, beta=1, gamma=0.1,
                 replay_alpha=0.9):

        self.task = Task(init_pose, init_velocities,
                         init_angle_velocities, runtime, target_pos)

        self.state_size = self.task.state_size
        self.action_size = self.task.action_size

        self.state = self.task.reset()

        self.memory = PrioritizedReplay(buffer_size, batch_size, replay_alpha)

        self.actor = Actor(self.state_size, self.action_size)
        self.actor_weights = self.actor.model.trainable_weights
        self.actor_target = Actor(self.state_size, self.action_size)

        self.critic = Critic(self.state_size, self.action_size)
        self.critic_weights = self.critic.model.trainable_weights
        self.critic_target = Critic(self.state_size, self.action_size)

        self.gamma = 0.1

        self.tau = 0.5

        self.training_step = 0

    def step(self):
        next_state = self.act()

        if self.memory.current_memory > self.memory.batch_size:
            self.learn()

        self.state = next_state

    def act(self):
        action = self.actor.model.predict(np.reshape(self.state, [-1, self.state_size]))
        next_state, reward, done = self.task.step(action)
        q = self.critic_target.model.predict(self.state, action)
        q_h = self.critic_target.model.predict(next_state, action)
        td = reward + self.gamma * q_h - q
        value = abs(td)
        self.memory.add(self.state, action, reward, next_state, done, value)
        return next_state

    def learn(self):
        self.training_step += 1

        experiences, weights = zip(*self.memory.sample())
        weights = self.memory.adjusted_weight(weights, self.training_step)
        states = np.array([experience.state for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        next_actions = self.actor_target.model.predict_on_batch(states)
        q_h = self.critic_target.model.predict_on_batch([next_states, next_actions])
        q = rewards - self.gamma * q_h * ( 1-dones)

        self.critic.model.train_on_batch(x=[states,actions],y=q,sample_weight=weights)

        gradients = self.critic.get_action_gradients(states, next_actions)

        self.actor.train_fn(states, gradients)

        self.target_update(self.actor, self.actor_target)
        self.target_update(self.critic, self.critic_target)

    def target_update(self, model, target_model):
        local_weights = np.array(model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def play(self):
        print('hi')

    def update_target(self, target):
        self.task.new_target(target)

    def save_model(self,name):
        print('hi')

    def load_model(self,name):
        print('hi')