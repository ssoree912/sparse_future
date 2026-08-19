from opencompass.models import Sparse_dLLM_LLaDACausalLM


# No eviction: the block cache is still frozen at step 1, but nothing is dropped.
# Isolates cache staleness from eviction capacity.
models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-sparse_dllm-keep1.0',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=1.0,
        scaling_config={},
        diffusion_config=dict(steps=512, block_length=32),
        seed=2025,
        model_type='llada',
        max_seq_len=2048,
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
