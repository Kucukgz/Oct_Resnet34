"""This prevents docstring warnings..."""
import os
import numpy as np
import PIL
import skimage
import torch
import torchvision
import pandas as pd
import visdom
import shutil
import random
import sklearn.metrics
import matplotlib
from PIL import Image
from skimage.io import imread

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# images
DATASET_ROOT = 'all'
RESIZE = 224

# Vision score
report = 'Report.xlsx'
all_data = pd.read_excel(report)
all_data = pd.DataFrame(all_data)
vision_score = all_data['Pre-op Vision']

# learning
LEARNING_RATE = 0.00001
BATCH_SIZE = 1
NUM_WORKERS = 1
RATIO = 5

# visdom
VISDOM_HOST = 'http://localhost'
VISDOM_PORT = 8097
# VISDOM_ENV = DATASET_NAME
VISDOM_ENV = 'retina_train2d'

# device
# device = torch.device('cpu')
device = torch.device('cuda')

# image transformation
transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
    # torchvision.transforms.Resize(RESIZE),
    torchvision.transforms.RandomRotation(60),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


class DatasetIM(torch.utils.data.Dataset):
    """This prevents docstring warnings..."""

    def __init__(self, root):
        """To prevents docstring warnings..."""
        # load 3D images
        self.root = root
        self.dataset_3d = os.listdir(root)
        self.dataset_3d = sorted(self.dataset_3d, key=lambda v: v.upper())
        # get dataset name and vision score
        self.dataset_2d = []
        for i in range(0, len(self.dataset_3d)):
            im = Image.open(os.path.join(root + '/', self.dataset_3d[i]))
            sum_planes = im.n_frames
            for j in range(0, sum_planes):
                first = self.dataset_3d[i].split('.tif')[0]
                vis_score_index = (all_data['Original Filename'] == first).argmax()
                vis_score = all_data['Pre-op Vision'][vis_score_index]
                self.dataset_2d.append(str(self.dataset_3d[i]) + ',' + str(j) + ',' + str(vis_score))
        # drop off outliers from dataset_2d
        outliers = []
        with open("file.txt", "r") as f:
            for line in f:
                outliers.append(int(line.strip()))
        for ele in sorted(outliers, reverse=True):
            del self.dataset_2d[ele]
        # shuffle dataset
        random.shuffle(self.dataset_2d)
        # splits data into train and test
        n_test = len(self.dataset_2d) // RATIO
        n_train = len(self.dataset_2d) - n_test
        self.train_dataset = self.dataset_2d[:n_train]
        self.test_dataset = self.dataset_2d[n_train:]

    def __len__(self):
        """To prevents docstring warnings..."""
        return len(self.dataset)

    def __getitem__(self, index):
        """To prevents docstring warnings..."""
        image_name = self.dataset[index]
        # split name and vision score
        i = image_name.split(',')[0]  # image name
        j = image_name.split(',')[1]  # which plane
        z = image_name.split(',')[2]  # vision score
        # load image
        image_path = os.path.join(self.root, i)
        img = Image.open(image_path)
        img.seek(int(j))
        img_array = np.array(img)
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_gray, (224, 224))
        image_pil = PIL.Image.fromarray(img_resized)
        image = transform(image_pil)  # normalize, ...
        # load vision score
        vision = np.asarray([float(z)], dtype=np.float32)
        gt = torch.from_numpy(vision)
        return image, gt, image_name


# RUN
print('---------------------------------------------------')
print('RUN')

# visdom
vis = visdom.Visdom(server=VISDOM_HOST, port=VISDOM_PORT, env=VISDOM_ENV)
vis.line(X=np.array([0]), Y=np.array([[np.nan]]),
         win='Train Loss', opts=dict(xlabel='epoch', ylabel='Train Loss',
         legend=['Train Loss']))
vis.line(X=np.array([0]), Y=np.array([[np.nan]]),
         win='Test Loss', opts=dict(xlabel='epoch', ylabel='Test Loss',
         legend=['Test Loss']))
vis.line(X=np.array([0]), Y=np.array([[np.nan]]),
         win='Train Accuray', opts=dict(xlabel='epoch', ylabel='Train Accuray',
         legend=['Train Accuracy']))
vis.line(X=np.array([0]), Y=np.array([[np.nan]]),
         win='Test Accuracy', opts=dict(xlabel='epoch', ylabel='Test Accuracy',
         legend=['Test Accuracy']))

# Data
datasets = DatasetIM(DATASET_ROOT)
test_dataset = DatasetIM(DATASET_ROOT)


# Train Dataset loaders
datasets.dataset = datasets.train_dataset

train_loader = torch.utils.data.DataLoader(
    datasets, BATCH_SIZE, num_workers=NUM_WORKERS,
    shuffle=True, pin_memory=True)
print(f'> Size of training dataset {len(datasets.train_dataset)}')

# Test Dataset Loader
test_dataset.dataset = datasets.test_dataset
test_loader = torch.utils.data.DataLoader(test_dataset)
print(f'> Size of testing dataset {len(datasets.test_dataset)}')

# model
model = torchvision.models.resnet34(pretrained=True)
num_infea = model.fc.in_features
model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
model.fc = torch.nn.Linear(num_infea, 1)
model.to(device)

# train parameters
num_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.00001)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()


loss_list = []
train_acc = []
test_acc = []

total_step = len(train_loader)

# loop
for epoch in range(num_epochs):
    print('-' * 50)
    print('epoch:' + str(epoch) + ' / ' + str(num_epochs))
    i = 0
    loss_item = 0
    for image, gt, image_name in train_loader:
        image = image.to(device=device, dtype=torch.float)
        gt = gt.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, gt)
        loss_item += loss.item()
        loss.backward()
        optimizer.step()
        if i % 4000 == 0:
            print('epoch: {} - {} / {}'.format(epoch, i, total_step))
        i += 1
    loss_list.append(loss.item())
    print(f'Loss : {loss_item / len(train_loader)}')
    print('*' * 50)

    # train
    correct = correct1 = correct2 = correct3 = 0
    total = 0
    train_loss = 0
    with torch.no_grad():
        for data in train_loader:
            image, gt, image_name = data
            image = image.to(device=device, dtype=torch.float)
            gt = gt.to(device=device, dtype=torch.float)
            output = model(image)
            loss = criterion(output.data, gt)
            train_loss += loss.item()
            total += gt.size(0)
            correct += (int(output.data) == int(gt))
    acc = 100 * correct / total
    print('Train'.center(50, '*'))
    print(f'Train Loss: {train_loss / total} \t Train Accuracy : {acc}')
    print('*' * 50)
    train_acc.append(acc)
    # To visualize
    vis.line(X=np.array([epoch]), Y=np.array([[train_loss / total]]),
             win='Train Loss', opts=dict(xlabel='epoch', ylabel='Train Loss',
             legend=['Train Loss']), update='append')
    vis.line(X=np.array([epoch]), Y=np.array([[acc / 100]]),
             win='Train Accuray', opts=dict(xlabel='epoch', ylabel='Train Accuray',
             legend=['Train Accuracy']), update='append')

    # test
    correct = correct1 = correct2 = correct3 = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            image, gt, image_name = data
            image = image.to(device=device, dtype=torch.float)
            gt = gt.to(device=device, dtype=torch.float)
            output = model(image)
            loss = criterion(output.data, gt)
            test_loss += loss.item()
            total += gt.size(0)
            correct += (int(output.data) == int(gt))
    acc = 100 * correct / total
    print('Test'.center(50, '*'))
    print(f'Test Loss: {test_loss / total} \t First Test Accuracy : {acc}')
    test_acc.append(acc)
    print('*' * 50)
    # To visualize
    vis.line(X=np.array([epoch]), Y=np.array([[test_loss / total]]),
             win='Test Loss', opts=dict(xlabel='epoch', ylabel='Test Loss',
             legend=['Test Loss']), update='append')
    vis.line(X=np.array([epoch]), Y=np.array([[acc / 100]]),
             win='Test Accuracy', opts=dict(xlabel='epoch', ylabel='Test Accuracy',
             legend=['Test Accuracy']), update='append')


# visualize
fig, ax1 = plt.subplots()

plt.plot(loss_list, label="Loss", color="black")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc) / 100, label="Test Acc", color="green")
ax2.plot(np.array(train_acc) / 100, label="Train Acc", color="red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()
