from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.optimizers import Adam
import keras.backend as K

class Actor:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.build_actor(self.state_size, self.action_size)

    def build_actor(self, state_size, action_size):
        h1_size = 128
        h2_size = 64


        states = Input(shape=[state_size], name='states')
        h1 = Dense(h1_size, activation='relu', name='hidden1')(states)
        h2 = Dense(h2_size, activation='relu', name='hidden2')(h1)
        # relu to make the min zero, step function in task
        # has safety to reduce high inputs to max speed
        actions = Dense(action_size, activation='relu', name='output_actions')(h2)
        self.model = Model(inputs=states, outputs=actions)

        action_gradients = Input(shape=([self.action_size]), name='action_grads')
        loss = K.mean(-action_gradients * actions)
        optimizer = Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)



class Critic:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.build_critic(self.state_size, self.action_size)

    def build_critic(self, state_size, action_size, learning_rate=0.05):
        hl1 = 128
        hl2 = 64
        hl3 = 32

        states = Input(shape=[state_size])
        s_1 = Dense(hl1, activation='relu')(states)
        s_2 = Dense(hl2, activation='relu')(s_1)

        actions = Input(shape=[action_size])
        a_1 = Dense(hl1, activation='relu')(actions)
        a_2 = Dense(hl2, activation='relu')(a_1)

        h_3 = Add()([s_2, a_2])
        h_4 = Dense(hl3, activation='relu')(h_3)

        Q = Dense(1, activation='linear')(h_4)

        self.model = Model(inputs=[states, actions], outputs=Q)

        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

        action_gradients = K.gradients(Q, actions)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)







