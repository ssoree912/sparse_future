from opencompass.models import Sparse_dLLM_LLaDACausalLM


# Dynamic cache eviction controls.
kernel_size = 3
keep_ratio = 0.1
block_length = 32
steps = 512

models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-sparse_dllm-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=kernel_size,
        keep_ratio=keep_ratio,
        scaling_config={},
        diffusion_config=dict(steps=steps, block_length=block_length),
        seed=2025,
        model_type='llada',
        max_seq_len=2048,
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
