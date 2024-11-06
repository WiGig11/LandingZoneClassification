import os
import tqdm
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate

# 自定义MNIST数据集
class CustomMNIST(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy().astype('uint8')
        img = torchvision.transforms.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# 自定义collate函数
def custom_collate(batch):
    new_batch = []
    for items in batch:
        image, label = items
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        new_batch.append((image, label_tensor))
    return default_collate(new_batch)

# 定义模型结构（简单的CNN模型）
class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输入通道数为1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),  # 根据输入尺寸调整
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 测试函数
def test(model, testloader, device, phase='Test'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm.tqdm(testloader, desc=f'{phase}'):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {phase} images: {accuracy:.2f} %')
    return accuracy

# 训练函数
def train(device):
    # 数据增强和预处理
    rotation_transforms = transforms.RandomChoice([
        transforms.RandomRotation([0, 0]),
        transforms.RandomRotation([90, 90]),
        transforms.RandomRotation([180, 180]),
        transforms.RandomRotation([270, 270])
    ])
    transform = transforms.Compose([
        rotation_transforms,
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_custom = transforms.Compose([
        rotation_transforms,
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # -----------------------------
    # 第一阶段：训练字母过滤器
    # -----------------------------

    # 加载字母数据（标签为0）
    letters_dataset = ImageFolder(root="./letters", transform=transform_custom)
    letters_labels = [0] * len(letters_dataset)  # 标记为0（字母）

    # 加载数字数据（标签为1）
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    target_digits = [1, 2, 3]
    idx_train = (mnist_train.targets == target_digits[0]) | (mnist_train.targets == target_digits[1]) | \
                (mnist_train.targets == target_digits[2])
    mnist_data_train = mnist_train.data[idx_train]
    mnist_targets_train = torch.ones(len(mnist_data_train), dtype=torch.long)  # 标记为1（数字）
    custom_mnist_dataset_train = CustomMNIST(mnist_data_train, mnist_targets_train, transform=transform)

    # 合并数据集
    train_dataset_stage1 = ConcatDataset([letters_dataset, custom_mnist_dataset_train])
    train_loader_stage1 = DataLoader(dataset=train_dataset_stage1, batch_size=32, shuffle=True, collate_fn=custom_collate)

    # 定义第一阶段模型和损失函数
    letter_detector = Model(num_classes=2).to(device)
    criterion_stage1 = nn.CrossEntropyLoss()
    optimizer_stage1 = optim.Adam(letter_detector.parameters(), lr=0.001)

    # 训练第一阶段模型
    print("Training Letter Detector (Stage 1)")
    for epoch in range(5):  # 可根据需要调整训练轮数
        letter_detector.train()
        running_loss = 0.0
        for data in tqdm.tqdm(train_loader_stage1, desc=f'Epoch {epoch+1}/5'):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer_stage1.zero_grad()
            outputs = letter_detector(inputs)
            loss = criterion_stage1(outputs, labels)
            loss.backward()
            optimizer_stage1.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader_stage1):.4f}')

    # -----------------------------
    # 第二阶段：训练数字分类器
    # -----------------------------

    # 使用第一阶段模型过滤数字样本
    print("Preparing Data for Digit Classifier (Stage 2)")
    # 从MNIST数据集中加载数字1、2、3的全部数据
    mnist_train_all = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    idx_train_all = (mnist_train_all.targets == 1) | (mnist_train_all.targets == 2) | (mnist_train_all.targets == 3)
    mnist_data_train_all = mnist_train_all.data[idx_train_all]
    mnist_targets_train_all = mnist_train_all.targets[idx_train_all] - 1  # 标签调整为0,1,2
    custom_mnist_dataset_train_all = CustomMNIST(mnist_data_train_all, mnist_targets_train_all, transform=transform)

    # 创建数据加载器
    data_loader_stage2 = DataLoader(dataset=custom_mnist_dataset_train_all, batch_size=32, shuffle=True, collate_fn=custom_collate)

    # 定义第二阶段模型和损失函数
    digit_classifier = Model(num_classes=3).to(device)
    criterion_stage2 = nn.CrossEntropyLoss()
    optimizer_stage2 = optim.Adam(digit_classifier.parameters(), lr=0.001)

    # 训练第二阶段模型
    print("Training Digit Classifier (Stage 2)")
    for epoch in range(5):  # 可根据需要调整训练轮数
        digit_classifier.train()
        running_loss = 0.0
        for data in tqdm.tqdm(data_loader_stage2, desc=f'Epoch {epoch+1}/5'):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer_stage2.zero_grad()
            outputs = digit_classifier(inputs)
            loss = criterion_stage2(outputs, labels)
            loss.backward()
            optimizer_stage2.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(data_loader_stage2):.4f}')

    # -----------------------------
    # 测试阶段
    # -----------------------------

    # 测试数据准备
    print("Preparing Test Data")
    # 加载测试字母数据集（标签为0）
    letters_dataset_test = ImageFolder(root="./letters_test", transform=transform_custom)
    # 如果没有单独的测试集，可以拆分部分训练集作为测试
    letters_labels_test = [0] * len(letters_dataset_test)  # 标记为0（字母）

    # 加载测试数字数据集（标签为1）
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    idx_test = (mnist_test.targets == 1) | (mnist_test.targets == 2) | (mnist_test.targets == 3)
    mnist_data_test = mnist_test.data[idx_test]
    mnist_targets_test = torch.ones(len(mnist_data_test), dtype=torch.long)  # 标签为1（数字）
    custom_mnist_dataset_test = CustomMNIST(mnist_data_test, mnist_targets_test, transform=transform)

    # 合并测试数据集
    test_dataset_stage1 = ConcatDataset([letters_dataset_test, custom_mnist_dataset_test])
    test_loader_stage1 = DataLoader(dataset=test_dataset_stage1, batch_size=32, shuffle=False, collate_fn=custom_collate)

    # 测试第一阶段模型
    print("Testing Letter Detector (Stage 1)")
    test(letter_detector, test_loader_stage1, device, phase='Stage 1 Test')

    # 测试第二阶段模型
    print("Testing Digit Classifier (Stage 2)")
    # 第二阶段测试数据
    data_loader_stage2_test = DataLoader(dataset=custom_mnist_dataset_test, batch_size=32, shuffle=False, collate_fn=custom_collate)
    test(digit_classifier, data_loader_stage2_test, device, phase='Stage 2 Test')

    # 保存模型
    print('Finished Training and Testing')
    PATH = 'models'
    os.makedirs(PATH, exist_ok=True)
    torch.save(letter_detector.state_dict(), os.path.join(PATH, 'letter_detector.pth'))
    torch.save(digit_classifier.state_dict(), os.path.join(PATH, 'digit_classifier.pth'))

# 主函数
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train(device)
