import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1=nn.Linear(inputSize,hiddenSize)
        self.linear2=nn.Linear(hiddenSize,outputSize)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        return x
    
    def Save(self,fileName='model.pth'):
        modelFolderPath='./model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        fileName=os.path.join(modelFolderPath,fileName)
        torch.save(self.state_dict(),fileName)

class QTrainer:

    def __init__(self,model,learningRate,gamma):
        self.learningRate=learningRate
        self.gamma=gamma
        self.model=model
        self.optimizer=optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion= nn.MSELoss()

    def TrainingStep(self, state, action, reward, nextState, done):
        state=torch.tensor(state, dtype=torch.float)
        nextState=torch.tensor(nextState, dtype=torch.float)
        action=torch.tensor(action, dtype=torch.float)
        reward=torch.tensor(reward, dtype=torch.float)
        #(n,x)

        if len(state.shape)==1:
            #(1,x)
            state=torch.unsqueeze(state, 0)
            nextState=torch.unsqueeze(nextState, 0)
            action=torch.unsqueeze(action, 0)
            reward=torch.unsqueeze(reward, 0)
            done=(done, )
        
        #1: predicted Q values with current state
        pred = self.model(state)

        target=pred.clone()
        for index in range(len(done)):
            QNew=reward[index]
            if not done[index]:
                QNew=reward[index]+self.gamma*torch.max(self.model(nextState[index]))
            target[index][torch.argmax(action).item()]=QNew
        
        self.optimizer.zero_grad()
        loss=self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()


        #2:QNew = r + y * max(nextPredicted Q value) -> only do this if not done
        #pred.clone()
        #preds[argmax(action)]=QNew


        
