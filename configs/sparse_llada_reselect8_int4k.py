from opencompass.models import Sparse_dLLM_LLaDACausalLM


# Re-selection with the retained pool compressed: V lives in host memory and the
# keys are int4. Keys only order candidates, so they tolerate precision the
# attention operands would not.
models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-reselect8-int4k-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=0.1,
        scorer='masked_row',
        reselect_every=8,
        reselect_offload_v=True,
        reselect_k_bits=4,
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
