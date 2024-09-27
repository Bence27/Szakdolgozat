from PIL import Image
import numpy as np
import gym
import os

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation,Convolution2D,Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


env=gym.make("snake:snake-v0", render_mode="human",sleep=0.1)
nb_actions=env.action_space.n
IMG_SHAPE=(84,84)
WINDOW_LENGTH=4

class ImageProcessor(Processor):
    def process_observation(self, observation):
        img=Image.fromarray(observation)
        img=img.resize(IMG_SHAPE)
        img=img.convert("L")
        img=np.array(img)
        return img.astype('uint8')
    
    def process_state_batch(self, batch):
        processed_batch=batch.astype('float32')/255.0
        return processed_batch
    
    def process_reward(self, reward):
        return np.clip(reward,-1.0,1.0)
 
 
class CustomModelIntervalCheckpoint(ModelIntervalCheckpoint):
    def __init__(self, filepath, interval, verbose=0):
        super(CustomModelIntervalCheckpoint, self).__init__(filepath, interval, verbose)
        self.a = 0

    def on_step_end(self, step, logs={}):
        self.a += 1  
        if self.a != 0 and self.a % self.interval == 0:
            # Create a unique checkpoint filename using the current step number
            filename = f"{self.filepath.split('.h5')[0]}_{self.a}.h5"
            if self.verbose > 0:
                print(f"\nStep {self.a}: saving model to {filename}")
            self.model.save_weights(filename, overwrite=True)
    
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
##print(model.summary())

memory=SequentialMemory(limit=1000000,window_length=WINDOW_LENGTH)
processor=ImageProcessor()
policy=LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=1.0,value_min=0.1,value_test=0.05,nb_steps=1000000)
#policy=LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=0.2,value_min=0.1,value_test=0.05,nb_steps=1000000)

dqn=DQNAgent(model=model,
             nb_actions=nb_actions,
             policy=policy,
             memory=memory,
             processor=processor,
             nb_steps_warmup=50000,
             gamma=.99,
             target_model_update=10000,
             train_interval=4,
             delta_clip=1)

dqn.compile(Adam(learning_rate=0.00025),metrics=['mae'])
weights_filename='DQN_BO.h5'
checkpoint_callback = CustomModelIntervalCheckpoint(filepath='DQN_CHECKPOINT_SNAKE.h5', interval=10000, verbose=0)
model.load_weights("dqn_snake_weights_ULT.h5")
#dqn.fit(env,nb_steps=20000,callbacks=[checkpoint_callback],log_interval=10000,visualize=False)
#dqn.save_weights('dqn_snake_weights_ULT.h5',overwrite=True)
dqn.test(env,nb_episodes=1,visualize=False)
