import os
import configparser
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

model_path = config.get('TRAINING', 'model_name')

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

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# 加载模型
model = CNN(num_classes=8)
model.load_state_dict(torch.load(model_path))
model.eval()

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        class_idx = predicted.item()
    return class_idx

# 测试
if __name__ == '__main__':
    image_path = 'path_to_your_image.png'  # 替换为你要预测的图像路径
    class_idx = predict_image(image_path)
    class_names = ['clap', 'kick', 'pickup', 'run', 'sitdown', 'standup', 'walk', 'wavehand']
    print(f'Predicted class: {class_names[class_idx]}')
