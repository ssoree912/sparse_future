from opencompass.models import Sparse_dLLM_LLaDACausalLM


# 확정된 실제 토큰이 보는 미래 어텐션을 'max'로 집계.
models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-oracle-confirmed-max-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=0.1,
        oracle_eviction=True,
        oracle_pool=True,
        oracle_rows='confirmed',
        oracle_aggregate='max',
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
