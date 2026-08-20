from opencompass.models import Sparse_dLLM_LLaDACausalLM


# Deployable re-selection: the step-1 K/V are retained and re-ranked with the
# current step's queries every 8 steps. No oracle, no recomputation.
models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-reselect8-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=0.1,
        scorer='masked_row',
        reselect_every=8,
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
