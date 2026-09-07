import torch

vector = torch.load("./vectors/sycophancy_response_avg_diff.pt")

print("形状:", vector.shape)
print("数据类型:", vector.dtype)
print()
print(vector)
