## copy from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

from typing import Optional
from .modeling_llada import CustomCache
@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             cache_scorer=None, student_question_window=128,
             student_score_head='attention', student_evict_suffix=False,
             oracle_eviction=False, oracle_pool=True, oracle_per_step=False,
             oracle_reselect_every=1, oracle_future_window=False,
             oracle_commit_rows=False, oracle_rows='masked', reselect_every=0,
             reselect_offload_v=False, reselect_k_bits=0,
             scorer='sparse_dllm'):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''  
    # print(steps, block_length)
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt.clone()
    
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    def row_weights_for(mode, was_mask, committed, confidence, just_revealed=None):
        if mode == 'answer':                       # AR과 동일: 확정된 토큰이 정확히 한 번 투표
            return just_revealed.to(confidence.dtype), False
        """Who votes at this step, and how much."""
        if mode == 'commit':                       # 이번 스텝 확정되는 행, confidence 가중
            return torch.where(committed, confidence, torch.zeros_like(confidence)), False
        if mode == 'confirmed':                    # 이미 확정된 행 전부, 행 평균
            return (~was_mask).to(confidence.dtype), True
        if mode == 'union':                        # 확정되는 행(confidence) ∪ 확정된 행(1.0)
            weights = torch.where(committed, confidence, torch.zeros_like(confidence))
            return weights + (~was_mask).to(confidence.dtype), True
        return was_mask.to(confidence.dtype), False   # 'masked': 아직 안 벗겨진 행

    for num_block in range(num_blocks):
        ## Initialize CustomCache for each block
        customcache = CustomCache(n_layers = model.config.n_layers, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                        kernel_size=model.config.kernel_size, keep_ratio=model.config.keep_ratio,
                        cache_scorer=cache_scorer, prompt_length=prompt_len,
                        generation_length=gen_length,
                        question_window=student_question_window,
                        score_head=student_score_head,
                        evict_suffix=student_evict_suffix)
        customcache.reselect_every = reselect_every
        customcache.offload_v = reselect_offload_v
        customcache.k_bits = reselect_k_bits
        customcache.scorer = scorer

        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        if oracle_eviction and not oracle_per_step:
            customcache.oracle_mode = 'record'

        def run_step(i):
            # Determine cache state: 0 for step 0, 1 for step 1, 2 for step 2+
            cache_state = 2 if i > 1 else i

            if cache_state != 2:
                model_input = x
                mask_index = (x == mask_id)
            else:
                model_input = x[:, block_start:block_end]
                mask_index = (model_input == mask_id)
            
            position_offset = block_start
            logits = model(model_input, position_offset, cache_state, customcache).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
            
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            if cache_state != 2:
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                return (transfer_index[0, block_start:block_end],
                        x0_p[0, block_start:block_end].detach())
            else:
                x0_block = torch.where(mask_index, x0, x[:, block_start:block_end])
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0_block, dtype=torch.bool, device=x0_block.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[:, block_start:block_end][transfer_index] = x0_block[transfer_index]
                return transfer_index[0], x0_p[0].detach()

        if oracle_per_step:
            # Re-select the kept 10% at every step from the attention that step
            # actually uses: run the step against the full cache to read it off,
            # undo the commit, then replay the same step against the pruned cache.
            customcache.oracle_mode = 'record'
            run_step(0)
            run_step(1)
            customcache.snapshot_full()
            i = 2
            while i < steps:
                window = min(oracle_reselect_every, steps - i)
                # Pass 1: run the window against the full cache and record what it
                # wants. With oracle_future_window the whole window is recorded, so
                # the selection is made from attention that has not happened yet —
                # each row votes only while it is still masked, i.e. right up to the
                # step that reveals it. Without the flag only the first step of the
                # window is recorded, which is present information measured under an
                # unpruned cache rather than a look-ahead.
                customcache.restore_full()
                x_before = x.clone()
                recorded = window if oracle_future_window else 1
                customcache.capture_rows = True
                customcache.set_row_mask(None)
                just_revealed = (x[0, block_start:block_end] != mask_id)
                for j in range(i, i + recorded):
                    was_mask = (x[0, block_start:block_end] == mask_id)
                    committed, confidence = run_step(j)
                    weights, normalize = row_weights_for(
                        oracle_rows, was_mask, committed, confidence, just_revealed)
                    customcache.fold_rows(weights, normalize)
                    just_revealed = committed
                customcache.capture_rows = False
                x.copy_(x_before)
                customcache.set_row_mask(None)
                customcache.apply_oracle(
                    model.config.keep_ratio,
                    pool_kernel=model.config.kernel_size if oracle_pool else None,
                )
                # Pass 2: replay the same window against the pruned cache.
                for j in range(i, i + window):
                    run_step(j)
                i += window
        elif reselect_every:
            # Practical re-selection: one forward per step; at every R-th step the
            # layer re-ranks the retained step-1 entries with its current queries.
            for i in range(steps):
                customcache.reselect_now = (i > 1 and (i - 2) % reselect_every == 0)
                # Step 1 builds the cache and makes the opening selection, so it
                # needs the row mask too.
                if customcache.reselect_now or i == 1:
                    customcache.set_row_mask(x[0, block_start:block_end] == mask_id)
                run_step(i)
            customcache.reselect_now = False
        elif not oracle_eviction:
            for i in range(steps):
                if i == 1:
                    customcache.set_row_mask(x[0, block_start:block_end] == mask_id)
                run_step(i)
        else:
            # Pass 1: full cache, record what this block actually attends to.
            run_step(0)
            run_step(1)
            x_snapshot = x.clone()
            if oracle_rows == 'final':
                # 답변을 full 캐시로 끝까지 만들고, 완성된 블록 상태에서 forward를 한 번 더
                # 돌려 그 답변이 실제로 참고한 것을 읽는다. 이때는 32행 전부가 실제 토큰이다.
                customcache.oracle_mode = None
                for i in range(2, steps):
                    run_step(i)
                customcache.oracle_mode = 'record'
                customcache.capture_rows = True
                customcache.set_row_mask(None)
                _, confidence = run_step(steps - 1)
                customcache.fold_rows(torch.ones_like(confidence), normalize=False)
                customcache.capture_rows = False
                x.copy_(x_snapshot)
                customcache.apply_oracle(
                    model.config.keep_ratio,
                    pool_kernel=model.config.kernel_size if oracle_pool else None,
                )
                for i in range(2, steps):
                    run_step(i)
                continue
            customcache.capture_rows = True
            customcache.set_row_mask(None)
            just_revealed = (x[0, block_start:block_end] != mask_id)
            for i in range(2, steps):
                was_mask = (x[0, block_start:block_end] == mask_id)
                committed, confidence = run_step(i)
                weights, normalize = row_weights_for(
                    oracle_rows, was_mask, committed, confidence, just_revealed)
                customcache.fold_rows(weights, normalize)
                just_revealed = committed
            customcache.capture_rows = False
            # Pass 2: replay the same steps against the oracle-pruned cache.
            x.copy_(x_snapshot)
            customcache.set_row_mask(None)
            customcache.apply_oracle(
                model.config.keep_ratio,
                pool_kernel=model.config.kernel_size if oracle_pool else None,
            )
            for i in range(2, steps):
                run_step(i)

    return x


def main():
    device = 'cuda'

    path = 'GSAI-ML/LLaDA-8B-Instruct'

    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
