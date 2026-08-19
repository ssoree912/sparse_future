"""Validate the oracle two-pass replay against the normal generation path."""
import sys, torch, inspect
sys.path.insert(0, '/workspace/dllm/opencompass')
from transformers import AutoConfig, AutoTokenizer
from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM, CustomCache
from opencompass.models.sparse_dllm.llada_generate import generate

print('CustomCache from:', inspect.getfile(CustomCache), flush=True)

path = '/workspace/dllm/model/LLaDA-8B-Instruct'
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
cfg.block_len = 32
cfg.kernel_size = 3
cfg.keep_ratio = 1.0
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = LLaDAModelLM.from_pretrained(path, config=cfg, device_map='auto',
                                     torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

text = ("Summarize the dialogue.\n\nAmanda: I baked cookies. Do you want some?\n"
        "Jerry: Sure!\nAmanda: I'll bring you tomorrow :-)\n\nSummary:")
ids = tok(text, return_tensors='pt').input_ids.to(model.device)
kw = dict(gen_length=64, block_length=32, steps=64, temperature=0., cfg_scale=0.,
          remasking='low_confidence')

plain = generate(model, ids, **kw)
oracle_full = generate(model, ids, oracle_eviction=True, oracle_pool=True, **kw)
print('keep=1.0 oracle replay matches plain path:',
      torch.equal(plain, oracle_full), flush=True)

model.config.keep_ratio = 0.1
base01 = generate(model, ids, **kw)
orac01 = generate(model, ids, oracle_eviction=True, oracle_pool=True, **kw)
print('keep=0.1 oracle differs from heuristic:', not torch.equal(base01, orac01), flush=True)
print('--- heuristic keep0.1 ---')
print(tok.decode(base01[0, ids.shape[1]:], skip_special_tokens=True))
print('--- oracle keep0.1 ---')
print(tok.decode(orac01[0, ids.shape[1]:], skip_special_tokens=True))
