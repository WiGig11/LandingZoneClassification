#DL lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate
from torch.utils.data import ConcatDataset,Dataset

#libs
import argparse
import logging
from termcolor import colored
import tqdm
import pdb
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

class CustomMNIST(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        image = TF.to_pil_image(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long) 
    
    @property
    def num_classes(self):
        return len(set(self.targets.numpy()))
    
    @property
    def classes(self):
        return set(self.targets.numpy())


def custom_collate(batch):
    new_batch = []
    for items in batch:
        image, label = items
        # 假设 'x' 对应的数值标签为 4，可以根据需要调整
        if label == 'x':
            label_tensor = torch.tensor(0, dtype=torch.long)
        else:
            label_tensor = torch.tensor(int(label), dtype=torch.long)
        new_batch.append((image, label_tensor))
    return default_collate(new_batch)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #pdb.set_trace()#
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def imshow(img,predicted):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.text(10, -30, str(predicted))
    plt.savefig('testres.jpg')

def test(model,testloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(testloader, 0)):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels+1).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    with torch.no_grad():
        inputs, labels = images.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    print(' '.join(f'{labels[j]}' for j in range(12)))
    print(predicted+1)
    imshow(torchvision.utils.make_grid(images),predicted)
    pass


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
target_digits = [1, 2, 3]
idx_train = (mnist_train.targets == target_digits[0]) | (mnist_train.targets == target_digits[1]) | \
        (mnist_train.targets == target_digits[2]) 
mnist_data_train = mnist_train.data[idx_train]
mnist_targets_train = mnist_train.targets[idx_train]
custom_mnist_dataset_train = CustomMNIST(mnist_data_train, mnist_targets_train, transform=transform)

mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
target_digits = [1, 2, 3]
idx = (mnist_test.targets == target_digits[0]) | (mnist_test.targets == target_digits[1]) | \
        (mnist_test.targets == target_digits[2]) 
mnist_data_test = mnist_test.data[idx]
mnist_targets_test = mnist_test.targets[idx]
custom_mnist_dataset_test = CustomMNIST(mnist_data_test, mnist_targets_test, transform=transform)

transform_custom = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将单通道转换为三通道
    transforms.ToTensor(),
    transforms.Resize((28, 28)), 
    transforms.Normalize((0.5,), (0.5,))
    # 你可能还需要其他转换，例如 Resize, Normalize 等
])
#transforms.Lambda(lambda x: x.point(lambda p: 255 if p > 0 else 0)),
emnist_train = ImageFolder(root="./data2", transform=transform_custom)
emnist_test = ImageFolder(root="./testdata2", transform=transform_custom)

train_dataset = torch.utils.data.ConcatDataset([custom_mnist_dataset_train, emnist_train])
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=12, shuffle=True,collate_fn=custom_collate)

test_dataset = torch.utils.data.ConcatDataset([custom_mnist_dataset_test, emnist_test])
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=16, shuffle=True,collate_fn=custom_collate)

emnist_test_match = ImageFolder(root="./123X", transform=transform_custom)
test_loader_match = torch.utils.data.DataLoader(dataset = emnist_test_match, batch_size=16, shuffle=True,collate_fn=custom_collate)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(colored(device, 'blue', attrs=['bold']))
#model = Model()
PATH = str('models')
PATH = os.path.join(PATH,'model.pt')
#model.load_state_dict(torch.load(PATH))
#model.to(device)
model = torch.load('models/model.pth')
model.eval()
test(model,test_loader_match,device)