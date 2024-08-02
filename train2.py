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
import matplotlib as plt
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 25)
        self.fc4 = nn.Linear(25, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #pdb.set_trace()#
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

def test(model,testloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(testloader, 0)):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #pdb.set_trace()
            correct += (predicted == labels+1).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the test images: {accuracy} %')
    pass

def train(device):
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    rotation_transforms = transforms.RandomChoice([
        transforms.RandomRotation([0, 0]),  # 旋转90度
        transforms.RandomRotation([90, 90]),  # 旋转90度
        transforms.RandomRotation([180, 180]),  # 旋转180度
        transforms.RandomRotation([270, 270])  # 旋转270度
    ])

    transform = transforms.Compose([
        rotation_transforms,
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])
        # 加载MNIST数据集并筛选出0, 1, 2, 3
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
        rotation_transforms,
        transforms.Grayscale(num_output_channels=1),  # 将单通道转换为三通道
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
        # 你可能还需要其他转换，例如 Resize, Normalize 等
    ])
    #transforms.Lambda(lambda x: x.point(lambda p: 255 if p > 0 else 0)),
    emnist_train = ImageFolder(root="./data_x", transform=transform_custom)
    emnist_test = ImageFolder(root="./testdata", transform=transform_custom)

    train_dataset = torch.utils.data.ConcatDataset([custom_mnist_dataset_train, emnist_train])
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=12, shuffle=True,collate_fn=custom_collate)

    test_dataset = torch.utils.data.ConcatDataset([custom_mnist_dataset_test, emnist_test])
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=12, shuffle=False,collate_fn=custom_collate)

    emnist_test_match = ImageFolder(root="./123X/123X", transform=transform_custom)
    test_loader_match = torch.utils.data.DataLoader(dataset = emnist_test_match, batch_size=16, shuffle=True,collate_fn=custom_collate)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    print("Training!")
    for epoch in range(int(10)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_loader),desc=f'Epoch = {epoch+1}/{10}'):
            #inputs, labels = data# get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            #pdb.set_trace()
            optimizer.zero_grad()# zero the parameter gradients
            outputs = model(inputs)# forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()            # print statistics
            #writer.add_scalar('Loss/train', loss, epoch)
        #if epoch%2==0:
        #    val(model,valloader,device,writer,epoch)

    print('Finished Training')
    test(model,test_loader_match,device)
    PATH = str('models')
    PATH = os.path.join(PATH,'model.pth')
    #torch.save(model.state_dict(), PATH)
    torch.save(model, PATH)
    pass

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(colored(device, 'blue', attrs=['bold']))
    #writer = SummaryWriter(os.path.join('runs',config.tensorboardname))    
    train(device)

        
if __name__ =='__main__':
    main()