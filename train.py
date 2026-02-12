
import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from src.datasets.defect_dataset import DefectDataset
from src.models.cnn_classifier import CNNClassifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DefectDataset("data/processed/train")
    val_dataset = DefectDataset("data/processed/val")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = CNNClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0

    for epoch in range(30):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("Best model saved.")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Validation accuracy: {acc:.3f}")
    return acc

if __name__ == "__main__":
    train()
