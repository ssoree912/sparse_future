# future_dllm

디퓨전 LLM의 KV 캐시 축출을, 완성된 답변이 무엇을 필요로 할지로 순위를 매겨 결정한다.

모델: `GSAI-ML/LLaDA-8B-Instruct`

## 설치

```bash
conda env create -f environment.yml
conda activate dllm
```

명령어는 저장소 루트에서 실행한다. 저장소 밖에 있다고 가정하는 경로는 둘이고, 둘 다 바꿀 수 있다:

| | 기본값 | 변경 |
|---|---|---|
| 모델 | `../model/LLaDA-8B-Instruct` | `FUTURE_DLLM_MODEL` 또는 `--model` |
| LongBench parquet | `../data/eval/longbench` | `LONGBENCH_DATA` |

## 데이터셋

| `--dataset` | 소스 (test 아닌 split) | `--gen-length` | 평가 task |
|---|---|---|---|
| `samsum` | LongBench samsum 대화 | 128 | `longbench_samsum` |
| `gsm8k` | gsm8k train | 128 | `gsm8k`, 5-shot |
| `mmlu` | mmlu validation + dev | 64 | `mmlu_generative` |
| `mbpp` | mbpp validation + prompt | 128 | `mbpp` |
| `math` | hendrycks_math train | 256 | `minerva_math` |
| `mbpp_full` | mbpp full train | 256 | `mbpp` |
| `samsum_lb` / `trec_lb` / `wiki2_lb` | LongBench 형식, ~2048토큰 | 128 / 96 / 64 | `longbench_*` |

`--gen-length`는 task의 `max_gen_toks`와 맞춰야 한다. 이 값이 캐시에서 프롬프트와 모델 출력의
비율을 결정하기 때문이다. 데이터셋 추가는 `teacher/build_prompt_shards.py`에 builder를 쓴다.

## 교사 라벨

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py     --dataset samsum --n-samples 300 --gen-length 128
```

→ `artifacts/prompt_shards/<dataset>/`, `artifacts/teacher/<dataset>/`. 둘 다 이어받으며,
개수를 올리면 없는 것만 추출한다.

## 학습

```bash
python student/train_student.py --teacher-root artifacts/teacher/samsum
```

교사 루트를 쉼표로 여러 개 주면 도메인 혼합 학습. `--epochs`, `--lr`, `--name`.
→ `artifacts/ckpts/<n>ds_<샘플수>_e<epoch>_lr<lr>_<해시>/`, `meta.json` 포함.

## 추론

```bash
scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]

scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
scripts/run_eval.sh gsm8k  1.0        # 축출 없음, 체크포인트 불필요
```

→ `results/<데이터셋>_keep<비율>_<날짜시간>.json`. `keep_ratio`가 1.0 미만이면 체크포인트가
필수. 문항 수는 `LIMIT` (기본 200). 생성 길이·정지 문자열·shot 수는 lm-eval task 정의에서
그대로 가져온다. LongBench task는 `eval/tasks/longbench/`에 있고 나머지는 lm-eval 기본이다.

생성 없이 held-out 라벨로 확인:

```bash
python student/eval_recall.py --student artifacts/ckpts/<이름>/checkpoint-best
```
