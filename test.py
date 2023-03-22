#!/usr/bin/env python3

import os
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


class FashionDataset(Dataset):
    def __init__(self, root_dir, txt_file,transform):
        self.root_dir = root_dir
        self.img_names, self.labels = self.load_data(txt_file)
        self.transform = transform
        #print(self.img_names)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        image = self.transform(image)
        return image, label

    def load_data(self, txt_file):
        with open(txt_file, 'r') as f:
            img_names = [line.strip() for line in f.readlines()]

        attr_file = os.path.splitext(txt_file)[0] + '_attr.txt'
        with open(attr_file, 'r') as f:
            labels = [[int(x) for x in line.strip().split()] for line in f.readlines()]

        return img_names, labels

transform = transforms.Compose([
    transforms.Resize((300, 200)),
    transforms.ToTensor()
])

train_dataset = FashionDataset('/home/delta/NTU/CV/codes/FashionDataset/FashionDataset/', '/home/delta/NTU/CV/codes/FashionDataset/FashionDataset/split/train.txt',transform)
val_dataset = FashionDataset('/home/delta/NTU/CV/codes/FashionDataset/FashionDataset/', '/home/delta/NTU/CV/codes/FashionDataset/FashionDataset/split/val.txt',transform)
list_attr = /home/delta/NTU/CV/codes/FashionDataset/FashionDataset/split
print(len(train_dataset),len(val_dataset))

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
print(len(train_loader),len(val_loader))

#Load label names with indexss  

# To check a single random image
trainiter = iter(train_loader)
img, label = next(trainiter)
index = 0
image = img[index]
print(type(image))
# Convert tensor to numpy array
img_array = image.numpy()
# Show image using plt.imshow
plt.imshow(img_array.transpose(1, 2, 0))
print(label[index].tolist())
plt.show()

# define the model
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# initialize the model and optimizer
model = CustomResNet50(num_classes=6)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# define the loss function
criterion = nn.BCEWithLogitsLoss()

# define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# move the model and loss function to the device
model = model.to(device)
criterion = criterion.to(device)

# # train the model
# for epoch in range(10):
#     train_loss = 0.0
#     train_acc = 0.0
#     val_loss = 0.0
#     val_acc = 0.0

#     model.train()
#     for inputs, targets in train_loader:
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * inputs.size(0)
#         train_acc += accuracy_score(targets.cpu().numpy(), (outputs.sigmoid().detach().cpu().numpy() > 0.5).astype(int))

#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs = inputs.to(device)
#             targets = targets.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             val_loss += loss.item
