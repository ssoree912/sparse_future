"""Confirm every mode retains the same number of cache entries per layer."""
import sys, json, torch
torch.set_grad_enabled(False)
sys.path.insert(0, '/workspace/dllm/opencompass')
from transformers import AutoConfig, AutoTokenizer
from opencompass.models.sparse_dllm import modeling_llada, llada_generate

MODES = [('baseline', dict(scorer='sparse_dllm')),
         ('ours', dict(scorer='masked_row')),
         ('oracle-masked', dict(oracle_eviction=True, oracle_rows='masked')),
         ('oracle-answer', dict(oracle_eviction=True, oracle_rows='answer')),
         ('oracle-final', dict(oracle_eviction=True, oracle_rows='final')),
         ('oracle-confirmed', dict(oracle_eviction=True, oracle_rows='confirmed'))]

path = '/workspace/dllm/model/LLaDA-8B-Instruct'
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
cfg.block_len, cfg.kernel_size, cfg.keep_ratio = 32, 3, 0.25
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = modeling_llada.LLaDAModelLM.from_pretrained(
    path, config=cfg, device_map='auto', torch_dtype=torch.bfloat16,
    trust_remote_code=True).eval()

sample = json.loads(open('/workspace/dllm/data/eval/math500/test_balanced100.jsonl').readline())
text = tok.apply_chat_template([{'role':'user','content':sample['problem']}],
                               add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)

seen = {}
original = modeling_llada.CustomCache.get_cache
def watched(self, layer_id):
    entry = original(self, layer_id)
    if entry.get('k') is not None:
        seen.setdefault(id(self), {})[layer_id] = entry['k'].shape[-2]
    return entry
modeling_llada.CustomCache.get_cache = watched

print(f'프롬프트 {ids.shape[1]}토큰, gen 64, block 32 → 후보 {ids.shape[1]+64-32}개, '
      f'기대 유지 {int((ids.shape[1]+64-32)*0.25)}개')
for name, kw in MODES:
    seen.clear()
    llada_generate.generate(model, ids, steps=64, gen_length=64, block_length=32,
                            temperature=0., cfg_scale=0., remasking='low_confidence', **kw)
    sizes = [s for per_cache in seen.values() for s in per_cache.values()]
    layers = [len(per_cache) for per_cache in seen.values()]
    print(f'{name:18s} 레이어 {min(layers)}~{max(layers)}개 관측 | '
          f'유지 크기 {min(sizes)}~{max(sizes)} (최빈 {max(set(sizes), key=sizes.count)})')
