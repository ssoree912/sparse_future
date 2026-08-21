"""Train a scorer to predict the final x row-max label.

Features are replayed rather than stored: each record keeps the exact block input
(`x_at_block_start`), so the forward that produced the student's deployment-time
hidden states can be reproduced here. That is the one thing the previous student
got wrong — it trained on a prompt-only forward and was deployed on a full
prompt+generation forward.

Targets are ranking targets, so the loss is listwise (KL against the normalised
label) plus a pairwise term sampled across the whole range, and checkpoints are
selected on mean recall over a k-grid rather than any single budget.
"""

from __future__ import annotations

import argparse, glob, json, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/workspace/dllm/model/LLaDA-8B-Instruct")
    p.add_argument("--teacher-root", default="/workspace/dllm/dLLM_f/results/budget/"
                                             "teacher/samsum",
                   help="comma-separated for mixed-domain training: val is split "
                        "per domain and the checkpoint is chosen on the domain "
                        "macro average, so a block-heavy domain cannot own it")
    p.add_argument("--output-dir", default="/workspace/dllm/dLLM_f/results/budget/"
                                           "student/samsum")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--question-window", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def recall_grid(pred, target, ratios=(0.05, 0.1, 0.2, 0.3, 0.5)):
    out = []
    for r in ratios:
        k = max(1, int(target.numel() * r))
        a = set(torch.topk(pred, k).indices.tolist())
        b = set(torch.topk(target, k).indices.tolist())
        out.append(len(a & b) / k)
    return sum(out) / len(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transformers import AutoConfig
    from future_dllm import LLaDAModelLM, CustomCache

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.block_len, cfg.kernel_size, cfg.keep_ratio = 32, 3, 1.0
    model = LLaDAModelLM.from_pretrained(args.model, config=cfg, device_map="auto",
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device, L, H = model.device, model.config.n_layers, model.config.d_model

    # Capture has to stay on so training sees the same hidden states deployment
    # will hand the scorer.
    CustomCache.capture_layer_hidden_states = (
        lambda self, layer_id, hidden: self.layer_hidden_states.__setitem__(layer_id, hidden))

    # Split val per domain: mmlu carries 2 blocks per sample against 4 elsewhere,
    # so a sample-balanced val would let the block-heavy domains own the choice.
    roots = [r for r in args.teacher_root.split(",") if r]
    train_shards, val_shards, datasets = [], [], []
    for root in roots:
        name = Path(root).name
        found = sorted(glob.glob(f"{root}/*.pt"))
        if not found:
            raise SystemExit(f"no teacher shards under {root}")
        split = max(1, int(len(found) * args.val_ratio))
        val_shards += [(name, p) for p in found[:split]]
        train_shards += [(name, p) for p in found[split:]]
        datasets.append(name)
        print(f"  {name}: train {len(found)-split} / val {split} shards", flush=True)
    print(f"train {len(train_shards)} / val {len(val_shards)} shards "
          f"over {len(datasets)} domain(s)", flush=True)

    # Same class the deployment path loads, so the checkpoint drops straight in.
    from future_dllm import PromptUtilityStudent, StudentConfig
    student_cfg = StudentConfig(layer_count=L, hidden_dim=H, proj_dim=args.proj_dim,
                                mlp_dim=args.mlp_dim, heads=("score",))
    student = PromptUtilityStudent(student_cfg).to(device).float()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def features(record):
        x = record["x_at_block_start"].unsqueeze(0).to(device)
        cache = CustomCache(n_layers=L, device=device, kernel_size=3, keep_ratio=1.0)
        cache.layer_hidden_states = {}
        model(x, int(record["block_start"]), 1, cache)
        return cache.layer_hidden_states

    def step(record, train: bool):
        hidden = features(record)
        cand = record["candidate_indices"].to(device)
        prompt_len = int(record["prompt_length"])
        ctx = torch.arange(max(0, prompt_len - args.question_window), prompt_len, device=device)
        label = record["label_final_rowmax"].float().to(device)
        total, recalls = 0.0, []
        for l in range(L):
            h = hidden[l].float()
            pred = student.forward_layer(l, h, cand, ctx, head="score").squeeze(0)
            tgt = label[l]
            if not torch.isfinite(tgt).all() or tgt.sum() <= 0:
                continue
            # listwise: KL against the normalised label distribution
            loss = F.kl_div(F.log_softmax(pred, -1), tgt / tgt.sum(), reduction="batchmean")
            # pairwise: random pairs anywhere in the range, to fix the ordering
            i = torch.randint(0, tgt.numel(), (args.pairs,), device=device)
            j = torch.randint(0, tgt.numel(), (args.pairs,), device=device)
            sign = torch.sign(tgt[i] - tgt[j])
            keep = sign != 0
            if keep.any():
                loss = loss + F.softplus(-sign[keep] * (pred[i][keep] - pred[j][keep])).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += float(loss.detach())
            recalls.append(recall_grid(pred.detach(), tgt))
        return total / max(1, L), sum(recalls) / max(1, len(recalls))

    best = -1.0
    for epoch in range(args.epochs):
        student.train(); random.shuffle(train_shards)
        started, losses = time.time(), []
        for n, (_, path) in enumerate(train_shards):
            for record in torch.load(path, map_location="cpu", weights_only=False)["blocks"]:
                losses.append(step(record, True)[0])
            if (n + 1) % 30 == 0:
                print(f"  epoch {epoch} {n+1}/{len(train_shards)} "
                      f"loss {sum(losses[-120:])/max(1,len(losses[-120:])):.4f}", flush=True)
        student.eval()
        per_ds = {}
        with torch.no_grad():
            for name, p in val_shards:
                for r in torch.load(p, map_location="cpu", weights_only=False)["blocks"]:
                    per_ds.setdefault(name, []).append(step(r, False)[1])
        means = {k: sum(v) / max(1, len(v)) for k, v in per_ds.items()}
        score = sum(means.values()) / max(1, len(means))   # domain macro average
        detail = "  ".join(f"{k} {v:.3f}" for k, v in sorted(means.items()))
        print(f"epoch {epoch}: loss {sum(losses)/len(losses):.4f} | "
              f"val recall macro {score:.4f} [{detail}] | "
              f"{(time.time()-started)/60:.1f}min", flush=True)
        if score > best:
            best = score
            ckpt = out_dir / "checkpoint-best"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in student.state_dict().items()},
                       ckpt / "pytorch_model.bin")
            json.dump({"layer_count": L, "hidden_dim": H, "proj_dim": args.proj_dim,
                       "mlp_dim": args.mlp_dim, "heads": ["score"]},
                      open(ckpt / "config.json", "w"))
            json.dump({"val_recall": score, "val_recall_per_dataset": means,
                       "epoch": epoch, "datasets": datasets,
                       "question_window": args.question_window},
                      open(out_dir / "best.json", "w"))
            print(f"  saved (best {best:.4f})", flush=True)
    print(f"done. best val recall {best:.4f} -> {out_dir}/checkpoint-best", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
