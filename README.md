# 基于WiFi CSI 的人体行为识别检测

## 项目概述

### 1. 文件结构

数据被组织在 `train` 和 `test` 目录中，每个目录包含不同动作类别的子目录：`clap`、`kick`、`pickup`、`run`、`sitdown`、`standup`、`walk`、`wavehand`，用于标识不同的动作

### 2. 项目原理

本项目使用卷积神经网络（CNN）来识别图像中的动作类别。主要步骤包括：

1. **数据预处理**：将图像调整为统一大小，并进行归一化处理。
2. **模型训练**：使用训练数据训练CNN模型，并在测试数据上进行验证。
3. **模型保存**：将训练好的模型保存为 `.pth` 文件。
4. **模型预测**：加载保存的模型，对新的图像进行分类预测。

### 3. 代码结构

- **train.py**：训练模型的脚本，包含数据加载、模型定义、训练和验证等功能。
- **predict.py**：加载保存的模型，并对新的图像进行分类预测。
- **config.ini**：配置文件，用于设置训练集和测试集路径、是否保存模型、模型文件名等参数。
- **requirements.txt**：项目依赖包列表，便于快速安装所需的Python包。
- **README.md**：项目说明文件。

## 使用方法

### 1. 安装依赖

**Python**版本：3.10,并确保安装了pytorch环境

```bash
pip install -r requirements.txt
```

### 2. 训练模型

在 `config.ini` 文件中配置训练集和测试集的路径、是否保存模型以及模型文件名。然后运行 `train.py` 脚本开始训练：

```bash
python train.py
```

### 3. 预测

在 `config.ini` 文件中配置保存的模型文件路径。然后运行 `predict.py` 脚本进行预测：

```bash
python predict.py --image_path <path_to_image>
```

## 数据生成

可以使用本项目自带的trans.py进行自动转换数据，会将采集CSI信号得到的dat文件转换为PNG图片，便于后续分析。

### 使用方法

``` bash
python trans.py
```

*注意在文件中修改文件路径，图片会保存在dat文件的目录中。

