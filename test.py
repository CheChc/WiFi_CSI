import torch

# 检查GPU是否可用
if torch.cuda.is_available():
    print("GPU可用！")
else:
    print("GPU不可用，将使用CPU进行计算。")

if torch.backends.mps.is_available():
    print("mps可用")
import torch
import timeit
import random

x = torch.ones(50000000,device='mps')
print(timeit.timeit(lambda:x*random.randint(0,100),number=1))