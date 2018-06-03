from collections import namedtuple
from operator import attrgetter
import numpy as np

class PrioritizedReplay:
    """
    Tested, no issues.
    """

    def __init__(self, buffer_size, batch_size, alpha):

        self.memory = []
        self.current_memory = 0
        self.min_value = 9999999999

        self.a = alpha
        self.beta_limit = 100000
        self.e = 0.001

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'value'])

    def add(self, state, action, reward, next_state, done, value):

        value += self.e

        if value <= self.min_value and len(self.memory) == self.buffer_size:
            return

        exp = self.experience(state, action, reward, next_state, done, value)

        if value > self.min_value and len(self.memory) == self.buffer_size:
            self.memory = sorted(self.memory, key=attrgetter('value'))
            self.memory[-1] = exp

        else:
            self.memory.append(exp)
            self.current_memory += 1
            if self.current_memory == self.buffer_size:
                attr = attrgetter('value')
                self.min_value = min([attr(e) for e in self.memory])

    def sample(self):
        attr = attrgetter('value')
        probs = np.array([attr(e) for e in self.memory])
        probs = (probs ** self.a) / (sum(probs) ** self.a)
        mem_prob = [(mem, probs) for (mem, probs) in zip(self.memory, probs)]
        # returns memory and corresponding probability
        return np.random.choice(mem_prob, size=self.batch_size, replace=True, p=probs)

    # todo sample weight conversion
    def adjusted_weight(self, probs, step_no):
        beta = min(1, step_no / self.beta_limit)
        probs = np.multiply(1 / probs, 1 / self.current_memory)
        probs = np.power(probs, beta)
        return probs
