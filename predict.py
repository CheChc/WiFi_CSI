import os
import configparser
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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
        self.fc1 = nn.Linear(64 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out

# 加载配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 获取配置参数
model_path = config['DEFAULT']['ModelPath']
image_path = config['DEFAULT']['ImagePath']
labels = config['DEFAULT']['Labels'].split(',')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载模型
model = CNN(num_classes=len(labels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 预测函数
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return labels[predicted.item()]

# 进行预测
predicted_class = predict(image_path)
print(f'预测类别是: {predicted_class}')
