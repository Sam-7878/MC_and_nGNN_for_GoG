import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def train_model(model, train_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        # Convert labels to Long type
        loss = criterion(out, data.y.long())  
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device):
    model.eval()
    preds, labels, probs = [], [], []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)

            prob = torch.softmax(out, dim=1)

            preds.extend(prob.argmax(dim=1).cpu().tolist())
            labels.extend(data.y.cpu().tolist())
            probs.extend(prob.cpu().numpy())  


    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    auc = roc_auc_score(labels, [p[1] for p in probs])  
    
    return accuracy, auc, f1
