from opencompass.models import Sparse_dLLM_LLaDACausalLM


# final x row-max 라벨로 학습한 scorer. 블록당 1회 선택, 보관 없음 —
# baseline과 같은 메모리 조건이고 미래 정보 없이 배포된다.
models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM,
        abbr='llada_8b_chat-student-final-rowmax-keep0.1',
        path='/workspace/dllm/model/LLaDA-8B-Instruct',
        kernel_size=3,
        keep_ratio=0.1,
        student_cache_path='/workspace/dllm/dLLM_f/results/budget/'
                           'student_final_rowmax_samsum300/checkpoint-best',
        student_score_head='score',
        student_question_window=128,
        student_evict_suffix=True,   # 라벨과 같은 후보 집합(프롬프트+suffix)
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
