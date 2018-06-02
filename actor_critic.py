from keras import Model
from keras.layers import Input, Dense, Add
from keras.optimizers import Adam

def make_actor(state_size, action_size):
    h1_size = 128
    h2_size = 64
    print('Building actor...')

    states = Input(shape=[state_size])
    h1 = Dense(h1_size, activation='relu')(states)
    h2 = Dense(h2_size, activation='relu')(h1)
    # relu to make the min zero, step function in task
    # has safety to reduce high inputs to max speed
    actions = Dense(action_size, activation='relu')(h2)
    model = Model(input=states, putput=actions)
    return model
    print('Actor built.')

def make_critic(state_size, action_size, learning_rate=0.05):
    hl1 = 128
    hl2 = 64
    hl3 = 32

    states = Input(shape=[state_size])
    s_1 = Dense(hl1, activation ='relu')(states)
    s_2 = Dense(hl2, activation='relu')(s_1)

    actions = Input(shape=[action_size])
    a_1 = Dense(hl1, activation ='relu')(actions)
    a_2 = Dense(hl2, activation='relu')(a_1)

    h_3 = Add()([s_2, a_2])
    h_4 = Dense(hl3, activation='relu')(h_3)

    Q = Dense(1, activation='linear')(h_4)

    model = Model(input=[states,actions], output=Q)

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    print('Critic compiled.')

    return model





