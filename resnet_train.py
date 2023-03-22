#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Load the pre-trained model
model = models.resnet50(pretrained=True)

# Remove the original classification layer
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(512, 26),
    nn.Sigmoid()
)

# Freeze the weights of the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Train the model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the performance on the validation set
with torch.no_grad():
    num_correct = 0
    num_samples = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        num_correct += (preds == labels).sum
