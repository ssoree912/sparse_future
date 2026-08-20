from opencompass.models import Sparse_dLLM_LLaDACausalLM


# 완성된 답변에서 forward 1회. 행 축을 max로 집계 — 한 답변 토큰이라도 강하게 본 후보를 살린다.
models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-oracle-final-rowmax-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=0.1,
        oracle_eviction=True,
        oracle_pool=True,
        oracle_rows='final',
        oracle_row_aggregate='max',
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
