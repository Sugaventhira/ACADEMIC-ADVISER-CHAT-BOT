#from ast import main
import json
from nltk_utils import tokenize,stem,list_of_words
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from model import NeuralNet

with open('tasks.json','r') as t:
    tasks = json.load(t)

allwords = []
tags = []
x_y = []

for task in tasks['tasks']:
    tag = task['tag']
    tags.append(tag)
    for pattern in task['ways']:
        w= tokenize(pattern)
        allwords.extend(w)
        x_y.append((w,tag))

ignore  = ['?',',','!',';',':','>','<','/','.']

allwords = [stem(w) for w in allwords if w not in ignore]
allwords = sorted(set(allwords))

x_train = []
y_train = []
for (pattern_s,tag) in x_y:
    list = list_of_words(pattern_s,allwords)
    x_train.append(list)

    label = tags.index(tag)
    y_train.append(label)

x_train=np.array(x_train)
y_train=np.array(y_train)

batch_size =8
input_size = len(x_train[0])
output_size = len(tags)
hidden_size = 8
learning_rate=0.001
num_epochs = 1000
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    #dataset[i]
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples
    


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size = batch_size,shuffle=True,num_workers=0)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words= words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100 == 0:
        print(f'i {epoch+1}/{num_epochs}, loss={loss.item():.4f}') 

print(f'final loss, loss={loss.item():.4f}') 


data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": allwords,
"tags": tags

}

FILE = "data.pth"
torch.save(data,FILE)

print("Training Finihed")

