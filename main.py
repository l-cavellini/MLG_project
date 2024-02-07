import torch
from model import HGT
from graph import load_graph
from torch_geometric.datasets import OGB_MAG
import pandas as pd 

def train(model, data, optimizer, criterion, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, val_mask):
    model.eval()
    with torch.no_grad():
        logits = model(data, val_mask)
        preds = logits.max(1)[1]
        acc = (preds == data['paper'].y[val_mask]).float().mean().item()
    return acc

if __name__ == '__main__':
    # retrieve the data
    dataset = OGB_MAG(root = 'data/amg', preprocess='metapath2vec')
    data = dataset[0]
    hid_dim = 128
    out_dim = dataset.num_classes
    num_epochs = 100
    model = HGT(graph = data, hidden_dim = hid_dim, out_dim = out_dim, n_heads = 8, n_layers = 1)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    val_accs = []
    

    best_val_acc = 0.0
    best_model = None

    for e in range(num_epochs):
        # training 
        train_losses.append(train(model, data, optimizer, criterion, data['paper'].train_mask))
        train_accs.append(evaluate(model, data, data.train_mask))
        
        # validation
        val_accs.append(evaluate(model, data, data.val_mask))
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_model = {key: value.cpu() for key, value in model.state_dict().items()}        
        print(f'Epoch: {e}, Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}')
    
    model.load_state_dict(best_model, 'best_model.pt')
    
    # saving the results
    pd.to_csv('results/train_losses.csv', train_losses)
    pd.to_csv('results/train_accs.csv', train_accs)
    pd.to_csv('results/val_accs.csv', val_accs)

    # testing the best model
    best_model = torch.load('path_to_save_model/best_model.pth')
    model.load_state_dict(best_model)
    test_acc = evaluate(model, data, data.test_mask)

    print(f'Test Acc: {test_acc:.4f}')

    