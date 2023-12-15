import random
import json
import torch
from model import NeuralNet
from nltk_utils import list_of_words,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('tasks.json','r') as t:
    tasks = json.load(t)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
allwords = data["all_words"]
tags = data['tags']
model_state = data["model_state"]


model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "TonyStark"

def get_response(msg):
    sentence = tokenize(msg)
    X = list_of_words(sentence, allwords)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for task in tasks['tasks']:
            if tag == task["tag"]:
                return random.choice(task['responses'])
   
    return " I couldn't get you... Better visit AR office for more informations "
   