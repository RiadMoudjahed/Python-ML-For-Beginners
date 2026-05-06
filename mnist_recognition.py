import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Convert MNIST Image Files into a Tensor of 4-Dimensions (# of images, Height, Width, Colors, Channels)
transform = transforms.ToTensor()

# Train Data
train_data = datasets.MNIST(root='./cnn_data/MNIST_data', train=True, download=True, transform=transform)
# Test Data
test_data = datasets.MNIST(root='./cnn_data/MNIST_data', train=False, download=True, transform=transform)
print ("="*10+"Train Data"+"="*10)
print (train_data)
print ("\n")

print ("="*10+"Test Data"+"="*10)
print (test_data)
print ("\n")

# Create DataLoaders for train and test data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_loader: # Loop through test_loader getting images, labels
        outputs = model(images) # Pass images through model to get outputs
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0) # Track total correct predictions and total samples
        correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%') # Print final accuracy as a percentage
    print ('\n')

all_preds = []
all_labels = []
for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    # After getting outputs from the model and predicted from torch.max, append them to your lists
    all_preds.extend(predicted.numpy()) # predicted digits
    all_labels.extend(labels.numpy()) # true digits

# Build the confusion matrix
cm = confusion_matrix(all_labels, all_preds) 
print(cm)

# Plot it
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
