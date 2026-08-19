from opencompass.models import Sparse_dLLM_LLaDACausalLM


student_checkpoint = (
    '/workspace/dllm/dLLM_f/results/budget/'
    'student_samsum300_dense_nochat_attention_delta_2head_'
    'rank0p1_topk0_e20_20260818/checkpoint-best'
)

models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-sparse_dllm-student-attn-topk50-suffix-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=0.1,
        student_cache_path=student_checkpoint,
        student_question_window=128,
        student_score_head='attention',
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
