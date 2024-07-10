import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import logging
import configparser

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 配置日志记录
logging.basicConfig(filename='test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = int(config['DEFAULT']['num_classes'])
model_path = config['DEFAULT']['model_path']

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

model = CNN(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 动作类别
classes = config['DEFAULT']['classes'].split(', ')

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return classes[predicted.item()]

def evaluate(testdata_dir):
    correct = 0
    total = 0
    for cls in os.listdir(testdata_dir):
        cls_path = os.path.join(testdata_dir, cls)
        for img_file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_file)
            predicted_class = predict_image(img_path)
            if predicted_class == cls:
                correct += 1
            total += 1
            logging.info(f'Image: {img_file}, Predicted: {predicted_class}, Actual: {cls}')
    accuracy = 100 * correct / total
    logging.info(f'Accuracy: {accuracy:.2f}%')
    print(f'Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    testdata_dir = config['DEFAULT']['testdata_dir']
    evaluate(testdata_dir)
