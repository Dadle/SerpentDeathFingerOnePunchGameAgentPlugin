# Keras for deep learning
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import RMSprop
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from serpent.input_controller import KeyboardKey, MouseButton
import numpy as np


class KerasDeepPlayer:

    # CNN-LSTM model parameters
    dropout_value = 0.2
    loss_function = 'categorical_crossentropy'  # 'mean_squared_error'
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=1e-6)
    moves = {'RIGHT': MouseButton.RIGHT, 'LEFT': MouseButton.LEFT, 'WAIT': None}
    moves_array = [MouseButton.RIGHT, MouseButton.LEFT, None]
    move_story = []

    def __init__(self, time_dim, game_frame_dim):
        self.model = Sequential()

        # First recurrent layer with dropout
        self.model.add(TimeDistributed(Convolution2D(filters=3, kernel_size=32, strides=(3, 3),
                                                     data_format="channels_last", border_mode='same', activation='relu',
                                                     W_constraint=maxnorm(3)), input_shape=time_dim + game_frame_dim))
        self.model.add(TimeDistributed(Dropout(self.dropout_value)))
        self.model.add(TimeDistributed(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3))))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Flatten()))

        # LSTM part of the network for handling time series and memory
        self.model.add(Bidirectional(LSTM(10, return_sequences=True)))
        self.model.add(Dropout(self.dropout_value))
        self.model.add(Bidirectional(LSTM(5, return_sequences=True)))
        self.model.add(Dropout(self.dropout_value))

        # Output layer (returns the predicted value)
        self.model.add(Dense(units=10, activation='tanh'))
        self.model.add(Dropout(self.dropout_value))
        self.model.add(Dense(units=3, activation='softmax'))

        # Set loss function and optimizer
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
        print("Built model for playing game:")
        print(self.model.summary())

    def decide(self, frame_buffer):
        move_per_timestep = list()
        frame_list = self.make_frame_list(frame_buffer)
        model_output = self.model.predict(frame_list)
        model_likelihood_time_array = model_output[0].tolist()#[len(model_output[0])-1]
        #print("Move likelihood:", model_likelihood_time_array)
        for model_likelihood_array in model_likelihood_time_array:
            max_likelihood = max(model_likelihood_array)
            move_index = model_likelihood_array.index(max_likelihood)
            #print("Selected move:", self.moves_array[move_index])
            move_per_timestep.append(self.moves_array[move_index])
        #self.move_story.append(move_per_timestep)
        return move_per_timestep

    def evaluate_move(self, move_per_timestep, frame_per_timestep, sequence_end_time, episode_start_time, episode_end_time):
        score_per_move = list()
        episode_score = episode_end_time - episode_start_time
        for i, move in enumerate(move_per_timestep):
            #print("move", move, "i", i)
            # decrease the sequence score as the time gets closer to the end. That is when the player is losing
            sequence_score = (episode_end_time - sequence_end_time) / episode_score

            one_hot_score = [0, 0, 0]
            #print("Move index", self.moves_array.index(move))
            one_hot_score[self.moves_array.index(move)] = sequence_score
            score_per_move.append(one_hot_score)
        return score_per_move

    def train_episode(self, episode, episode_start_time, episode_end_time):
        print("Training after completing an episode!")
        print("I survived for ", episode_end_time-episode_start_time, "seconds this time")
        for sequence in episode:
            frame_list = self.make_frame_list(sequence['frames'])
            scores = sequence['scores']
            #print("Got score:", score)
            return self.model.fit(x=frame_list, y=np.array([scores]), batch_size=1, epochs=1,
                                  verbose=1, shuffle=False)  #.history

    @staticmethod
    def make_frame_list(frame_buffer):
        frame_list = list()
        for game_frame in frame_buffer:
            print(game_frame.frame.shape)
            frame_list.append(game_frame.frame)
        if len(np.array(frame_list).shape) == 4:
            return np.array([frame_list])
        else:
            return frame_list

