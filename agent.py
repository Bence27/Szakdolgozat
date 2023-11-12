import torch
import random
import numpy as np
from collections import deque
from snake import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import Plot

MAX_MEMORY=100_000
BATCH_SIZE=1000
LR=0.001

class Agent:

    def __init__(self):
        self.numberOfGames=0
        self.epsilon=0 #randomness
        self.gamma=0.9 # discount rate must be <1
        self.memory=deque(maxlen=MAX_MEMORY) #popleft()
        self.model=Linear_QNet(11,256,3)
        self.trainer=QTrainer(self.model,learningRate=LR,gamma=self.gamma)

    def GetState(self,game):
        head=game.snake[0]
        pointLeft=Point(head.x-20,head.y)
        pointRight=Point(head.x+20,head.y)
        pointUp=Point(head.x,head.y-20)
        pointDown=Point(head.x,head.y+20)

        directionLeft=game.direction==Direction.LEFT
        directionRight=game.direction==Direction.RIGHT
        directionUp=game.direction==Direction.UP
        directionDown=game.direction==Direction.DOWN

        state=[
            #danger straight
            (directionRight and game.is_collision(pointRight)) or
            (directionLeft and game.is_collision(pointLeft)) or
            (directionUp and game.is_collision(pointUp)) or
            (directionDown and game.is_collision(pointDown)),

            #danger right
            (directionUp and game.is_collision(pointRight)) or
            (directionDown and game.is_collision(pointLeft)) or
            (directionLeft and game.is_collision(pointUp)) or
            (directionRight and game.is_collision(pointDown)),

            #danger left
            (directionDown and game.is_collision(pointRight)) or
            (directionUp and game.is_collision(pointLeft)) or
            (directionRight and game.is_collision(pointUp)) or
            (directionLeft and game.is_collision(pointDown)),

            #Move direction
            directionLeft,
            directionRight,
            directionUp,
            directionDown,

            #food location
            game.food.x<game.head.x,
            game.food.x>game.head.x,
            game.food.y<game.head.y,
            game.food.y>game.head.y
        ]
        return np.array(state, dtype=int)

    def Remember(self,state,action,reward,nextState,done):
        self.memory.append((state,action,reward,nextState,done)) #popleft if MAXMERORY is reached

    def TrainLongMemory(self):
        if len(self.memory)>BATCH_SIZE:
            miniSample=random.sample(self.memory, BATCH_SIZE) #list of tuples
        else:
            miniSample=self.memory
        states, actions, rewards, nextStates, dones=zip(*miniSample)
        self.trainer.TrainingStep(states, actions, rewards, nextStates, dones)

    def TrainShortMemory(self, state, action, reward, nextState, done):
        self.trainer.TrainingStep(state, action, reward, nextState, done)

    def GetAction(self,state):
        #random moves: tradeof exploration / expoitation
        self.epsilon=80-self.numberOfGames
        finalMove=[0,0,0]
        if random.randint(0,200)<self.epsilon:
            move=random.randint(0,2)
            finalMove[move]=1
        else:
            state0=torch.tensor(state, dtype=torch.float)
            prediction=self.model(state0)
            move=torch.argmax(prediction).item()
            finalMove[move]=1
        return finalMove


def Train():
    plotScores=[]
    plotMeanScores=[]
    totalScore=0
    record=0
    agent=Agent()
    game=SnakeGameAI()
    while True:
        #get old state
        stateOld=agent.GetState(game)

        #get move
        finalMove=agent.GetAction(stateOld)

        # perform move and get new state
        reward,done,score=game.play_step(finalMove)
        stateNew=agent.GetState(game)

        #train short memory
        agent.TrainShortMemory(stateOld, finalMove, reward, stateNew, done)

        #remember
        agent.Remember(stateOld, finalMove, reward, stateNew, done)

        if done:
            #train long memory
            game.Reset()
            agent.numberOfGames+=1
            agent.TrainLongMemory()

            if score>record:
                record=score
                agent.model.Save()
            
            print('Game',agent.numberOfGames, 'Score', score, 'Record',record)

            plotScores.append(score)
            totalScore+=score
            meanScore=totalScore/agent.numberOfGames
            plotMeanScores.append(meanScore)
            Plot(plotMeanScores,plotMeanScores)



if __name__=='__main__':
    Train()