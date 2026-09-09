import torch

vector_1 = torch.load("./vectors/讨好型/sycophancy_response_avg_diff.pt")
vector_2 = torch.load("./vectors/golden_gate_bridge_response_avg_diff.pt")
result_vector = vector_1+vector_2

torch.save(result_vector, "./vectors/goldenbridge_sycophancy_mix.pt")
print("反人格向量已保存，形状:", result_vector.shape)
