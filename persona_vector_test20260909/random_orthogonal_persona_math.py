import torch

vector = torch.load("./vectors/sycophancy_response_avg_diff.pt")  # [29, 3584]

def random_orthogonal(v, seed=0):
    """对每一层分别构造一个与v正交的随机方向"""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    # Gram-Schmidt：去掉r在v方向上的分量
    proj = (r * v).sum(dim=-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) ** 2 + 1e-8)
    r_orth = r - proj * v
    return r_orth

orth_vector = random_orthogonal(vector)
torch.save(orth_vector, "./vectors/random_orthogonal_to_sycophancy.pt")
