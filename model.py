from tensorflow import keras
from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Dense, Flatten, Activation,Convolution2D,Permute,Lambda,Input
from keras.optimizers import Adam

IMG_SHAPE=(84,84)
WINDOW_LENGTH=4

class Model:
    
    def build_model_atari(nb_actions):
        input_shape=(WINDOW_LENGTH, IMG_SHAPE[0],IMG_SHAPE[1])
        model=Sequential()
        model.add(Permute((2,3,1),input_shape=input_shape))
        model.add(Convolution2D(32,(8,8),strides=(4,4), kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64,(4,4),strides=(2,2), kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64,(3,3),strides=(1,1), kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        return model
    
    def build_dueling_model_atari(nb_actions):
        input_shape=(WINDOW_LENGTH, IMG_SHAPE[0],IMG_SHAPE[1])
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=input_shape))
        model.add(Convolution2D(32, (8, 8), strides=(4, 4), kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Flatten())
        value_fc = Dense(512, activation='relu')(model.output)
        value = Dense(1, activation='linear')(value_fc)
        advantage_fc = Dense(512, activation='relu')(model.output)
        advantage = Dense(nb_actions, activation='linear')(advantage_fc)
        q_values = Lambda(lambda args: args[0] + (args[1] - keras.backend.mean(args[1], axis=1, keepdims=True)))([value, advantage])
        model = KerasModel(inputs=model.input, outputs=q_values)
        return model
    
    def build_model_classic_control(obs_shape,nb_actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + obs_shape)) 
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(nb_actions, activation="linear"))
        return model

