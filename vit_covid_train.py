# -------------------- Standard Libraries --------------------
import os
import random
import argparse

# -------------------- Data Handling & Visualization --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Image Processing --------------------
from PIL import Image

# -------------------- PyTorch Core --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# -------------------- TorchVision --------------------
from torchvision import transforms
from torchvision.datasets import ImageFolder

# -------------------- Model Utilities --------------------
import timm

# -------------------- Evaluation Metrics --------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# -------------------- TensorFlow (only for version print) --------------------
import tensorflow as tf

def main():
    # -------------------- Setup and Reproducibility --------------------
    seed = 42
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    seed_everything(seed)

    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch version: {torch.__version__}, TensorFlow version: {tf.__version__}")

    # -------------------- Dataset and Transformations --------------------
    data_dir = os.environ.get('DATA_DIR','/mnt/home/sattum/vit_covid_xai/data/')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total images: {len(dataset)}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes: {dataset.classes}")

    # -------------------- Vision Transformer Model --------------------
    class ViTModel(nn.Module):
        def __init__(self, num_classes=2):
            super(ViTModel, self).__init__()
            self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        def forward(self, x):
            return self.vit(x)

    model = ViTModel(num_classes=2).to(device)

    # -------------------- Loss and Optimizer --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    # -------------------- Metric Function --------------------
    def compute_metrics(y_true, y_pred, y_prob, title="Confusion Matrix", show_cm=True, save_path=None):
        y_prob = np.array(y_prob)
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob[:, 1])

        if show_cm or save_path:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "COVID"], yticklabels=["Normal", "COVID"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(title)
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            if show_cm:
                plt.show()
            plt.close()

        return accuracy, precision, recall, f1, auc

    # -------------------- Training Loop --------------------
    num_epochs = 10
    best_val_accuracy = 0.0
    patience = 3
    patience_counter = 0
    best_model_state = None
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs)

        train_acc, _, _, _, _ = compute_metrics(all_labels, all_preds, all_probs, show_cm=False)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_labels, val_preds, val_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                predicted = torch.argmax(outputs, dim=1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(probs)

        val_acc, _, _, _, _ = compute_metrics(val_labels, val_preds, val_probs, show_cm=False)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "outputs/vit_covid_model.pth")
        print("Best model restored and saved.")

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/train_val_accuracy.png")
    plt.show()

    # Test
    model.eval()
    test_labels, test_preds, test_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted = torch.argmax(outputs, dim=1)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())
            test_probs.extend(probs)

    accuracy, precision, recall, f1, auc = compute_metrics(
        test_labels, test_preds, test_probs,
        title="Confusion Matrix (Test)", show_cm=True,
        save_path="outputs/confusion_matrix_test.png"
    )
    print(f"Test Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    metrics_df = pd.DataFrame([{
        "Test Accuracy (%)": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    }])
    metrics_df.to_csv("outputs/test_metrics.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT COVID Classification and XAI")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to image dataset')
    args = parser.parse_args()
    os.environ['DATA_DIR'] = args.data_dir
    main()
