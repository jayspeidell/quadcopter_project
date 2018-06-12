import numpy as np
from task import Task
from actor_critic import Actor, Critic, Noise
from replay import PrioritizedReplay
import sys
import keras.backend as K
import tensorflow as tf


class Agent:
    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None,
                 buffer_size=150000, batch_size=32, gamma=0.99,
                 replay_alpha=0.5, beta_limit=10000):

        self.task = Task(init_pose, init_velocities,
                         init_angle_velocities, runtime, target_pos)

        self.state_size = self.task.state_size
        self.action_size = self.task.action_size

        self.state = self.task.reset()

        self.memory = PrioritizedReplay(buffer_size, batch_size, replay_alpha, beta_limit)

        self.actor = Actor(self.state_size, self.action_size, self.task.action_low, self.task.action_high)
        self.actor_weights = self.actor.model.trainable_weights
        self.actor_target = Actor(self.state_size, self.action_size, self.task.action_low, self.task.action_high)

        self.critic = Critic(self.state_size, self.action_size)
        self.critic_weights = self.critic.model.trainable_weights
        self.critic_target = Critic(self.state_size, self.action_size)

        self.gamma = gamma

        # how much influence older weights have when updating target
        self.tau = 0.03

        #noise
        # GENTLE LANDING
        #self.mu = 0
        #self.theta = 0.1
        #self.sigma = 25
        self.mu = 0
        self.theta = 0.1
        self.sigma = 9
        self.noise = Noise(self.action_size, self.mu, self.theta, self.sigma)

        self.episodes = 0
        self.training_step = 0

    def step(self):
        done = 0
        while not done:
            next_state, done = self.act()
            loss = None
            if self.memory.current_memory > self.memory.batch_size:
                loss = self.learn()

            self.state = next_state

        # reset episode
        if self.training_step > 0:
            self.episodes += 1
        if self.episodes > 1 and self.episodes % 20 == 0 and loss:
            print('%d episode loss: %f' % (self.episodes, loss))
        self.state = self.task.reset()
        self.noise.reset()
        self.noise.reset()

    def act(self):
        # output is 2d array, convert to 1d with [0]
        action = self.actor.model.predict(np.reshape(self.state, [-1, self.state_size]))[0]
        action = self.noise.add_noise(action)
        action = self.task.clip(action)
        #if self.training_step % 250 == 0:
        #    print(action)
        next_state, reward, done = self.task.step(action)
        q = self.critic_target.model.predict([np.reshape(self.state, [-1, self.state_size]), np.reshape(action, [-1, self.action_size])])
        q_h = self.critic_target.model.predict([np.reshape(next_state, [-1, self.state_size]), np.reshape(action, [-1, self.action_size])])
        td = reward + self.gamma * q_h - q
        value = abs(float(td[0]))
        self.memory.add(self.state, action, reward, next_state, done, value)
        return next_state, done

    def learn(self):
        self.training_step += 1
        experiences, weights = zip(*self.memory.sample())
        experiences = list(experiences)
        weights = np.array(weights)
        weights = self.memory.adjusted_weight(weights, self.training_step)
        states = np.array([experience.state for experience in experiences])
        next_states = np.array([experience.next_state for experience in experiences])
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        dones = np.array([experience.done for experience in experiences])

        next_actions = self.actor_target.model.predict_on_batch(states)
        q_h = self.critic_target.model.predict_on_batch([next_states, next_actions])
        q = np.reshape(rewards, [-1, 1]) - self.gamma * q_h * (1-np.reshape(dones, [-1, 1]))

        loss = self.critic.model.train_on_batch(x=[states, actions], y=q, sample_weight=weights)

        # problem: it's a list with one item / band-aid: [0]
        gradients = self.critic.get_action_gradients([states, next_actions])[0]

        self.actor.train_fn([states, gradients])

        self.target_update(self.actor, self.actor_target)
        self.target_update(self.critic, self.critic_target)

        return loss

    def target_update(self, model, target_model):
        # todo can't get weights, test the quick fix you did before
        local_weights = np.array(model.model.get_weights())
        target_weights = np.array(target_model.model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.model.set_weights(new_weights)

    def play(self):
        step = 0
        state = self.task.reset()
        action = self.actor.model.predict(np.reshape(state, [-1, self.state_size]))[0]
        done = 0
        rewards = [0]
        distances = [0]
        positions = [state]
        actions = [action]
        speeds = [0]
        velocities = [[0, 0, 0, 0]]
        while not done:
            step += 1
            state, reward, done = self.task.step(action)
            if done:
                print('DONE!')
            action = self.actor.model.predict(np.reshape(state, [-1, self.state_size]))[0]
            distances.append(self.task.dist)
            rewards.append(reward)
            positions.append(state)
            actions.append(action)
            speeds.append(self.task.speed)
            velocities.append(self.task.sim.v)
            if step > 5000:
                break
        return positions, actions, rewards, distances, speeds

    def sample_play(self, action):
        step = 0
        state = self.task.reset()

        done = 0
        rewards = [0]
        distances = [0]
        positions = [state]
        actions = [action]
        speeds = [0]
        velocities = [[0, 0, 0, 0]]
        while not done:
            step += 1
            state, reward, done = self.task.step(action)
            #action = self.actor.model.predict(np.reshape(state, [-1, self.state_size]))[0]
            distances.append(self.task.dist)
            rewards.append(reward)
            positions.append(state)
            actions.append(action)
            speeds.append(self.task.speed)
            velocities.append(self.task.sim.v)
            if step > 5000:
                break
        return positions, actions, rewards, distances, speeds

    def update_target(self, target):
        self.task.new_target(target)

    def save_model(self,name):
        print('hi')

    def load_model(self,name):
        print('hi')