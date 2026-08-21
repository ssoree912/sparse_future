"""samsum으로만 학습한 scorer가 다른 도메인 라벨을 얼마나 맞추나.

학습을 돌리기 전에, 라벨이 정말 도메인 불변인지 값싸게 확인한다.
recall@k는 train_final_rowmax_student.py와 같은 정의(k∈{5,10,20,30,50}% 평균).
"""
import argparse, glob, random, sys, torch
from pathlib import Path


def recall_grid(pred, target, ratios=(0.05, 0.1, 0.2, 0.3, 0.5)):
    out = []
    for r in ratios:
        k = max(1, int(target.numel() * r))
        a = set(torch.topk(pred, k).indices.tolist())
        b = set(torch.topk(target, k).indices.tolist())
        out.append(len(a & b) / k)
    return sum(out) / len(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/workspace/dllm/model/LLaDA-8B-Instruct")
    p.add_argument("--student", default="/workspace/dllm/dLLM_f/results/budget/"
                                        "student_final_rowmax_samsum300/checkpoint-best")
    p.add_argument("--root", default="/workspace/dllm/dLLM_f/results/budget")
    p.add_argument("--datasets", default="samsum300,gsm8k,mmlu,mbpp")
    p.add_argument("--shards", type=int, default=25)
    p.add_argument("--question-window", type=int, default=128)
    p.add_argument("--opencompass-root", default="/workspace/dllm/opencompass")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    random.seed(0)
    sys.path.insert(0, args.opencompass_root)
    from transformers import AutoConfig
    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM, CustomCache
    from opencompass.models.sparse_dllm.student_cache import load_prompt_utility_student

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.block_len, cfg.kernel_size, cfg.keep_ratio = 32, 3, 1.0
    model = LLaDAModelLM.from_pretrained(args.model, config=cfg, device_map="auto",
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).eval()
    CustomCache.capture_layer_hidden_states = (
        lambda self, layer_id, hidden: self.layer_hidden_states.__setitem__(layer_id, hidden))
    device, L = model.device, model.config.n_layers

    student = load_prompt_utility_student(args.student, device).float().eval()

    def features(record):
        x = record["x_at_block_start"].unsqueeze(0).to(device)
        cache = CustomCache(n_layers=L, device=device, kernel_size=3, keep_ratio=1.0)
        cache.layer_hidden_states = {}
        model(x, int(record["block_start"]), 1, cache)
        return cache.layer_hidden_states

    print(f"{'dataset':10s} {'블록':>5s} {'student':>8s} {'무작위':>7s} {'최근성':>7s}")
    for ds in args.datasets.split(","):
        shards = sorted(glob.glob(f"{args.root}/teacher_final_rowmax_{ds}/*.pt"))[:args.shards]
        if not shards:
            print(f"{ds:10s} (shard 없음)")
            continue
        stu, rnd, rec, n = [], [], [], 0
        for path in shards:
            for record in torch.load(path, map_location="cpu", weights_only=False)["blocks"]:
                hidden = features(record)
                cand = record["candidate_indices"].to(device)
                prompt_len = int(record["prompt_length"])
                ctx = torch.arange(max(0, prompt_len - args.question_window), prompt_len,
                                   device=device)
                label = record["label_final_rowmax"].float().to(device)
                C = label.size(-1)
                for l in range(L):
                    tgt = label[l]
                    if not torch.isfinite(tgt).all() or tgt.sum() <= 0:
                        continue
                    pred = student.forward_layer(l, hidden[l].float(), cand, ctx,
                                                 head="score").squeeze(0)
                    stu.append(recall_grid(pred, tgt))
                    rnd.append(recall_grid(torch.randn(C, device=device), tgt))
                    rec.append(recall_grid(torch.arange(C, device=device).float(), tgt))
                n += 1
        m = lambda v: sum(v) / max(1, len(v))
        print(f"{ds:10s} {n:5d} {m(stu):8.3f} {m(rnd):7.3f} {m(rec):7.3f}", flush=True)


if __name__ == "__main__":
    main()
