import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torchvision.models import vit_b_16


def attention_similarity_metrics(attn_weights, values, outputs, include_cls=False):
    token_start = 0 if include_cls else 1
    v = values[..., token_start:, :]
    y = outputs[..., token_start:, :]
    a = attn_weights[..., token_start:, token_start:]

    seq_len = v.shape[2]
    if seq_len < 2:
        return {"value_cos": float("nan"), "diag_attn": float("nan"), "output_cos": float("nan")}

    value_norm = F.normalize(v, dim=-1)
    output_norm = F.normalize(y, dim=-1)

    pairwise = value_norm @ value_norm.transpose(-1, -2)
    upper = torch.triu(torch.ones(seq_len, seq_len, device=v.device, dtype=torch.bool), diagonal=1)
    avg_value_cos = pairwise[..., upper].mean().item()

    diag = a.diagonal(dim1=-2, dim2=-1)
    avg_diag = diag.mean().item()

    avg_output_cos = (output_norm * value_norm).sum(dim=-1).mean().item()
    return {"value_cos": avg_value_cos, "diag_attn": avg_diag, "output_cos": avg_output_cos}


def split_heads(x, num_heads):
    b, t, e = x.shape
    d = e // num_heads
    return x.view(b, t, num_heads, d).transpose(1, 2)


def make_random_image_batch(num_images=64, image_size=224, seed=2026):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((num_images, 3, image_size, image_size), generator=g)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def analyze_vit(model, images, batch_size=8, include_cls=False):
    model.eval()
    device = next(model.parameters()).device
    layers = list(model.encoder.layers)
    n_layers = len(layers)

    keys = ("value_cos", "diag_attn", "output_cos")
    sums = {k: [0.0] * n_layers for k in keys}
    total = 0

    with torch.no_grad():
        for s in range(0, images.shape[0], batch_size):
            xb = images[s : s + batch_size].to(device)

            x = model._process_input(xb)
            n = x.shape[0]
            cls = model.class_token.expand(n, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = model.encoder.dropout(x + model.encoder.pos_embedding)

            traces = {k: [] for k in keys}
            for block in layers:
                x_ln = block.ln_1(x)
                sa = block.self_attention

                qkv = F.linear(x_ln, sa.in_proj_weight, sa.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)
                q = split_heads(q, sa.num_heads)
                k = split_heads(k, sa.num_heads)
                v = split_heads(v, sa.num_heads)

                scores = (q @ k.transpose(-1, -2)) / math.sqrt(sa.head_dim)
                attn_w = torch.softmax(scores, dim=-1)
                y = attn_w @ v

                m = attention_similarity_metrics(attn_w, v, y, include_cls=include_cls)
                for kname in keys:
                    traces[kname].append(m[kname])

                attn_out = y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], sa.embed_dim)
                attn_out = F.linear(attn_out, sa.out_proj.weight, sa.out_proj.bias)
                attn_out = block.dropout(attn_out)
                x = x + attn_out

                y_ffn = block.ln_2(x)
                y_ffn = block.mlp(y_ffn)
                x = x + y_ffn

            bs = xb.shape[0]
            for kname in keys:
                for li, val in enumerate(traces[kname]):
                    sums[kname][li] += val * bs
            total += bs

    return {k: [v / max(total, 1) for v in sums[k]] for k in keys}


def plot_paper_metrics(metric_series, title, output_path):
    import matplotlib.pyplot as plt

    labels = [
        ("value_cos", r"Avg $\cos(v_i, v_j)$, $i<j$", "Cosine similarity"),
        ("diag_attn", r"Avg $a_{i,i}$", "Attention score"),
        ("output_cos", r"Avg $\cos(y_i, v_i)$", "Cosine similarity"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    layer_ids = list(range(len(metric_series["value_cos"])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, (key, panel_title, ylabel), color in zip(axes, labels, colors):
        ax.plot(layer_ids, metric_series[key], color=color, linewidth=2.3, marker="o", markersize=4)
        ax.set_title(panel_title)
        ax.set_xlabel("Layer index")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="ViT paper-metric diagnostics and plotting.")
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--random_init", action="store_true", help="Use random-init ViT instead of pretrained")
    p.add_argument("--include_cls", action="store_true", help="Include CLS token in metrics")
    p.add_argument("--device", default=None, help="cuda | cpu | mps (default auto)")
    p.add_argument("--output_plot", default="vit_paper_metrics.png")
    p.add_argument("--output_json", default="vit_paper_metrics.json")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    if args.image_size != 224:
        raise ValueError("This script targets torchvision ViT-B/16 and expects --image_size 224.")

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    weights = None if args.random_init else "DEFAULT"
    model = vit_b_16(weights=weights).to(device).eval()

    images = make_random_image_batch(
        num_images=args.num_images,
        image_size=args.image_size,
        seed=args.seed,
    )

    t0 = time.time()
    metrics = analyze_vit(
        model=model,
        images=images,
        batch_size=args.batch_size,
        include_cls=args.include_cls,
    )
    elapsed = time.time() - t0

    title = (
        f"ViT-B/16 ({'random-init' if args.random_init else 'pretrained'}) "
        f"- {args.num_images} images"
    )
    plot_paper_metrics(metrics, title, args.output_plot)

    payload = {
        "runtime_sec": round(elapsed, 3),
        "device": device,
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "include_cls": args.include_cls,
        "random_init": args.random_init,
        "output_plot": args.output_plot,
        "metrics": metrics,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved plot: {args.output_plot}")
    print(f"Saved metrics: {args.output_json}")


if __name__ == "__main__":
    main()
