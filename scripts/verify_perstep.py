"""Check the per-step oracle replay against the plain path at keep=1.0."""
import sys, torch
sys.path.insert(0, '/workspace/dllm/opencompass')
from transformers import AutoConfig, AutoTokenizer
from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM
from opencompass.models.sparse_dllm.llada_generate import generate

path = '/workspace/dllm/model/LLaDA-8B-Instruct'
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
cfg.block_len, cfg.kernel_size, cfg.keep_ratio = 32, 3, 1.0
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = LLaDAModelLM.from_pretrained(path, config=cfg, device_map='auto',
                                     torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

text = ("Summarize the dialogue into a short summary.\n\nDialogue:\nAmanda: I baked cookies. "
        "Do you want some?\nJerry: Sure!\nAmanda: I'll bring you tomorrow :-)\n\nSummary:")
ids = tok(text, return_tensors='pt').input_ids.to(model.device)
kw = dict(gen_length=64, block_length=32, steps=64, temperature=0., cfg_scale=0.,
          remasking='low_confidence')

plain = generate(model, ids, **kw)
per_full = generate(model, ids, oracle_eviction=True, oracle_per_step=True, **kw)
print('per-step oracle at keep=1.0 matches plain path:', torch.equal(plain, per_full), flush=True)

model.config.keep_ratio = 0.1
block01 = generate(model, ids, oracle_eviction=True, **kw)
per01 = generate(model, ids, oracle_eviction=True, oracle_per_step=True, **kw)
print('keep=0.1 per-step differs from block oracle:', not torch.equal(block01, per01), flush=True)
print('block  oracle:', tok.decode(block01[0, ids.shape[1]:], skip_special_tokens=True)[:120])
print('perstep oracle:', tok.decode(per01[0, ids.shape[1]:], skip_special_tokens=True)[:120])
