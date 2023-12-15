import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self,inp,hidden,num_class):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(inp ,hidden)
        self.l2 = nn.Linear(hidden ,hidden)
        self.l3 = nn.Linear(hidden ,num_class)
        self.relu = nn.ReLU()


    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out