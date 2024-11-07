# Import dependencies
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from smdebug.pytorch import Hook  # Import smdebug hook
import smdebug.pytorch as smd  # Import the smdebug library for debugging/profiling
from torchvision import datasets, models
from torch.utils.data import DataLoader
import argparse

# Data loading with transforms
def create_data_loaders(data_dir, batch_size=32):
    """
    Create data loaders for training, validation, and testing datasets.
    Includes resizing, normalization, and data augmentation for training.
    """
    # Transformations for training (with augmentation) and testing (only resize and normalization)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for ImageNet models
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=f"{data_dir}/valid", transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transforms)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

# Model initialization with a pretrained ResNet50
def net():
    """
    Initialize the model using a pretrained ResNet50 model with custom final layer.
    """
    model = models.resnet50(pretrained=True)

    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False   

    # Replace the final layer with a new layer for 5 classes (for bin item counts)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )
    return model

# Training function with validation accuracy included
def train(model, train_loader, validation_loader, criterion, optimizer, device, num_epochs, hook=None):
    model.to(device)
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if hook:
                hook.record_tensor_value("train_loss", loss)
                
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train

        # Validation loop
        if hook:
            hook.set_mode(smd.modes.EVAL)
        
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
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

        if hook:
            hook.set_mode(smd.modes.TRAIN)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {valid_loss/len(validation_loader):.4f}, "
              f"Validation Accuracy: {valid_accuracy:.2f}%")

# Test function for the test dataset
def test(model, test_loader, criterion, device, hook=None):
    model.eval()
    if hook:
        hook.set_mode(smd.modes.EVAL)
    
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if hook:
                hook.record_tensor_value("test_loss", loss)

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

    # Initialize the debugging/profiling hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(args.data, args.batch_size)

    # Training the model with hook
    train(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs, hook=hook)
    
    # Final evaluation on the test set with hook
    test(model, test_loader, criterion, device, hook=hook)

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    print("Model saved as model.pth")

# Command-line argument parsing
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 'data'), help='Directory for the dataset')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'), help='Directory to save the model')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '.'), help='Output data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    main(args)
