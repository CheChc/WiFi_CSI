import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import configparser
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 配置参数
data_dir_train = config.get('DATA', 'train_data_dir')
data_dir_test = config.get('DATA', 'test_data_dir')
model_save_path = config.get('TRAINING', 'model_save_path')
save_model = config.getboolean('TRAINING', 'save_model')
num_epochs = config.getint('TRAINING', 'num_epochs')
learning_rate = config.getfloat('TRAINING', 'learning_rate')
batch_size = config.getint('TRAINING', 'batch_size')
max_files_per_class = config.getint('DATA', 'max_files_per_class')
log_step = config.getint('TRAINING', 'log_step')

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 自定义数据集类
class LimitedImageFolder(Dataset):
    def __init__(self, root, transform=None, max_files_per_class=None):
        self.root = root
        self.transform = transform
        self.max_files_per_class = max_files_per_class
        self.classes = sorted(os.listdir(root))
        self.images = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root, cls)
            cls_files = os.listdir(cls_path)

            if self.max_files_per_class:
                cls_files = cls_files[:self.max_files_per_class]

            for file in cls_files:
                self.images.append(os.path.join(cls_path, file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 加载数据
train_dataset = LimitedImageFolder(root=data_dir_train, transform=transform, max_files_per_class=max_files_per_class)
test_dataset = LimitedImageFolder(root=data_dir_test, transform=transform, max_files_per_class=max_files_per_class)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out

model = CNN(num_classes=8).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model(num_epochs):
    train_losses = []  # 用于存储每个epoch的平均损失
    for epoch in range(num_epochs):
        logging.info(f'Starting epoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % log_step == 0:
                log_msg = f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}'
                print(log_msg)
                logging.info(log_msg)

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # 记录平均损失
        accuracy = 100 * correct / total
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] completed. Avg Train Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%')

        print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Avg Train Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if save_model:
        torch.save(model.state_dict(), model_save_path)
        logging.info(f'Model saved to {model_save_path}')

    # 绘制损失曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

train_model(num_epochs)
