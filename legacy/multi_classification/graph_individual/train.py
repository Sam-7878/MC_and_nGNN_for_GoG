import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def calculate_misclassification_rate(labels, preds, num_classes):
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    misclassification_rate = 1 - np.trace(cm) / np.sum(cm)
    return misclassification_rate

def train(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for data in loader:
        data = data.to(device)
        # print(data)

        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = out.argmax(dim=1)
        probs = torch.softmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    all_probs = np.array(all_probs)
    if all_probs.shape[1] < num_classes:
        all_probs = np.hstack([all_probs, np.zeros((all_probs.shape[0], num_classes - all_probs.shape[1]))])

    average_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    # misclassification_rate = calculate_misclassification_rate(all_labels, all_preds, num_classes)

    return average_loss, accuracy, precision, recall, f1_macro, f1_micro, # misclassification_rate

def evaluate(model, loader, criterion, device, num_classes, report = False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()

            preds = out.argmax(dim=1)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    all_probs = np.array(all_probs)

    if all_probs.shape[1] < num_classes:
        all_probs = np.hstack([all_probs, np.zeros((all_probs.shape[0], num_classes - all_probs.shape[1]))])

    average_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    if report:
        print(classification_report(all_labels, all_preds, zero_division=0))

    return average_loss, accuracy, precision, recall, f1_macro, f1_micro