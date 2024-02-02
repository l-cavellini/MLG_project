import torch
import torch.nn as nn
from model import HGT

def train(model, data, optimizer: str, criterion, num_epochs = 100, lr = 0.01, weight_decay = 0.01, print_loss = True):
        losses = []
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)   
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay)
        
        for e in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model.forward(data)
            loss = criterion(out, data.y)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        return loss, out

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)