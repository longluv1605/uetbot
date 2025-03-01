import torch
print(torch.cuda.is_available())  # Phải True
print(torch.cuda.get_device_name(0))  # RTX 3050