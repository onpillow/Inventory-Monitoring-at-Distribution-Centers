# Import dependencies
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import argparse

# Data loading with transforms
def create_data_loaders(data_dir, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader = DataLoader(datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(datasets.ImageFolder(root=f"{data_dir}/valid", transform=test_transforms), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transforms), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

# Model initialization with a pretrained ResNet50
def net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(inplace=True), nn.Linear(128, 5))
    return model

# Training function with validation accuracy included
def train(model, train_loader, validation_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train
        valid_loss, correct_valid, total_valid = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()
        valid_accuracy = 100 * correct_valid / total_valid
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Valid Loss: {valid_loss/len(validation_loader):.4f}, Valid Acc: {valid_accuracy:.2f}%")

# Test function for the test dataset
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    average_loss = total_loss / total
    accuracy = correct / total
    print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Main function to set up and start the training and evaluation process
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, valid_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    train(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    test(model, test_loader, criterion, device)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    print("Model saved as model.pth")

# Command-line argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 'data'), help='Directory for the dataset')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'), help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    main(args)
