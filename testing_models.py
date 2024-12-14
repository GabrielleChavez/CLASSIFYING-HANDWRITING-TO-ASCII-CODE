import numpy as np
import pandas as pd
import torch
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score


def evaluate_model(model_name, y_true, y_pred, get_ROC_plot=False):
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")
    if get_ROC_plot:
        plotROCcurve(fpr, tpr, auc)

    return f1, acc, cm, auc

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

    # Calculate F1 score and confusion matrix
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # evaluate_model(model, all_labels, all_preds, get_ROC_plot=True)
    return avg_loss, accuracy, f1, cm