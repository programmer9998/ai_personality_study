"""
compare_angle.py
------------------------------------------------------------
对比两个persona vector（比如 esfp_a 和 intj_t）之间的"夹角"，
逐层计算余弦相似度，并换算成角度（0°=完全同向，90°=完全无关，
180°=完全反向）。

用法:
  # 直接对比两个向量文件（29层，每层都会给出夹角）
  python compare_angle.py \
      --vector_a ./vectors/esfp_a_response_avg_diff.pt \
      --vector_b ./vectors/intj_t_response_avg_diff.pt

  # 如果想验证"esfp_a 是不是约等于 -intj_t"，加 --negate_b
  # 这样会先把vector_b取负号，再算夹角（理论上应该接近0°）
  python compare_angle.py \
      --vector_a ./vectors/esfp_a_response_avg_diff.pt \
      --vector_b ./vectors/intj_t_response_avg_diff.pt \
      --negate_b
------------------------------------------------------------
"""

import argparse
import torch


def cosine_and_angle(a: torch.Tensor, b: torch.Tensor):
    """
    逐层算余弦相似度和对应角度。
    a, b: [num_layers+1, hidden_dim]
    返回: cos_sim [num_layers+1], angle_deg [num_layers+1]
    """
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    # clamp一下，避免浮点误差导致 arccos 的输入超出 [-1, 1] 报NaN
    cos_sim_clamped = cos_sim.clamp(-1.0, 1.0)
    angle_rad = torch.arccos(cos_sim_clamped)
    angle_deg = angle_rad * 180.0 / torch.pi
    return cos_sim, angle_deg


def print_per_layer_table(cos_sim: torch.Tensor, angle_deg: torch.Tensor):
    print(f"\n{'层(layer)':>10} | {'余弦相似度':>10} | {'夹角(度)':>10} | 直观含义")
    print("-" * 60)
    for i in range(len(cos_sim)):
        c = cos_sim[i].item()
        a = angle_deg[i].item()
        if c > 0.7:
            hint = "方向高度一致"
        elif c > 0.2:
            hint = "有一定相关性"
        elif c > -0.2:
            hint = "基本无关(接近正交)"
        elif c > -0.7:
            hint = "有一定对立性"
        else:
            hint = "方向高度相反"
        print(f"{i:>10} | {c:>10.3f} | {a:>9.1f}° | {hint}")


def print_summary(cos_sim: torch.Tensor, angle_deg: torch.Tensor):
    print("\n[整体统计]")
    print(f"  平均余弦相似度: {cos_sim.mean().item():.3f}")
    print(f"  平均夹角: {angle_deg.mean().item():.1f}°")

    min_idx = angle_deg.argmin().item()
    max_idx = angle_deg.argmax().item()
    print(f"  夹角最小(最相似)的层: layer {min_idx}  ({angle_deg[min_idx].item():.1f}°)")
    print(f"  夹角最大(最不同)的层: layer {max_idx}  ({angle_deg[max_idx].item():.1f}°)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector_a", required=True)
    parser.add_argument("--vector_b", required=True)
    parser.add_argument("--negate_b", action="store_true",
                         help="对比前先把vector_b取负号（用于验证'A是否约等于-B'这类假设）")
    args = parser.parse_args()

    vec_a = torch.load(args.vector_a)
    vec_b = torch.load(args.vector_b)

    if args.negate_b:
        vec_b = -vec_b
        print("[提示] 已将 vector_b 取负号后再对比")

    name_a = args.vector_a.split("/")[-1]
    name_b = args.vector_b.split("/")[-1]
    print(f"\n对比: {name_a}  vs  {name_b}")
    print(f"形状: {tuple(vec_a.shape)}  vs  {tuple(vec_b.shape)}")

    cos_sim, angle_deg = cosine_and_angle(vec_a, vec_b)
    print_per_layer_table(cos_sim, angle_deg)
    print_summary(cos_sim, angle_deg)

    # 额外算一个"把所有层拉平成一个大向量"的整体夹角，作为补充视角
    flat_cos = torch.nn.functional.cosine_similarity(
        vec_a.flatten().unsqueeze(0), vec_b.flatten().unsqueeze(0)
    ).item()
    flat_angle = torch.arccos(torch.tensor(flat_cos).clamp(-1.0, 1.0)).item() * 180.0 / torch.pi
    print(f"\n[整体拉平后的夹角(所有层拼成一个大向量)]")
    print(f"  余弦相似度: {flat_cos:.3f}   夹角: {flat_angle:.1f}°")


if __name__ == "__main__":
    main()
