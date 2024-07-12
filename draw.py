import matplotlib.pyplot as plt
import re

# 读取日志文件
log_file_path = './log.txt'  # 请替换为你的日志文件路径
with open(log_file_path, 'r') as file:
    log_data = file.readlines()

# 提取 epoch, accuracy, 和 test accuracy
epochs = []
accuracies = []
test_accuracies = []

current_epoch = None

for line in log_data:
    # 检查 epoch 开始
    if "Starting epoch" in line:
        epoch_match = re.search(r'Starting epoch (\d+)/\d+', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

    # 提取训练 accuracy
    if "Avg Train Loss" in line:
        accuracy_match = re.search(r'Accuracy: ([\d.]+)%', line)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
            accuracies.append(accuracy)

    # 提取测试 accuracy
    if "Avg Test Loss" in line:
        test_accuracy_match = re.search(r'Test Accuracy: ([\d.]+)%', line)
        if test_accuracy_match:
            test_accuracy = float(test_accuracy_match.group(1))
            test_accuracies.append(test_accuracy)

# 确保 epochs, accuracies 和 test_accuracies 长度相同
min_length = min(len(epochs), len(accuracies), len(test_accuracies))
epochs = epochs[:min_length]
accuracies = accuracies[:min_length]
test_accuracies = test_accuracies[:min_length]

# 绘制 accuracy 和 test accuracy 的图像
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, label='Accuracy', color='blue', linestyle='-')
plt.plot(epochs, test_accuracies, label='Test Accuracy', color='red', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy and Test Accuracy over Epochs')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig('accuracy_vs_test_accuracy.png')
plt.close()
