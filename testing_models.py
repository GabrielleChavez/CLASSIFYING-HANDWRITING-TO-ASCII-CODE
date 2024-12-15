import numpy as np
import pandas as pd
import torch
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score


def evaluate_model(model_name, y_true, y_pred, get_ROC_plot=False):
    f1 = f1_score(y_true, y_pred, labels = range(0,94), average='weighted')
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels = range(0,94), average = 'weighted')
    recall = recall_score(y_true, y_pred, labels = range(0,94), average = 'weighted')
    return f1, acc, cm, prec, recall

def plotROCcurve(fpr, tpr, auc):
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def test_model(model, data_loader, device, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient computation for testing
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get class with max probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and true labels for F1 score and confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # Calculate average loss and accuracy
    avg_loss = test_loss / len(data_loader)
    accuracy = 100 * correct / total


    f1, acc, cm = evaluate_model(str(type(model)), all_labels, all_preds, get_ROC_plot=True)
    return avg_loss, accuracy, f1, cm