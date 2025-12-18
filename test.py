import torch

# 1. 检查 CUDA 是否可用（最关键的一步）
is_available = torch.cuda.is_available()
print(f"CUDA (GPU) 是否可用: {is_available}")

# 2. 如果可用，查看具体信息
if is_available:
    print(f"当前使用的 GPU 设备名: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch 编译的 CUDA 版本: {torch.version.cuda}")
    print(f"当前 GPU 数量: {torch.cuda.device_count()}")
else:
    print("当前使用的是 CPU 版本，或者未正确安装显卡驱动。")
