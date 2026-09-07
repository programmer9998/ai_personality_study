import torch

vector = torch.load("./vectors/sycophancy_response_avg_diff.pt")
anti_vector = -vector

torch.save(anti_vector, "./vectors/anti_sycophancy_response_avg_diff.pt")
print("反人格向量已保存，形状:", anti_vector.shape)
