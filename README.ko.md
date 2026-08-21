# future_dllm

디퓨전 LLM의 KV 캐시 축출을, 완성된 답변이 무엇을 필요로 할지로 순위를 매겨 결정한다.

## 환경

```bash
conda activate dllm          # torch 2.5.1+cu124, transformers 4.46.3, lm-eval 0.4.12
```

모델: [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), 로컬 사본은
`../model/LLaDA-8B-Instruct`. GPU 1장, batch size 1 — 캐시 상태가 시퀀스마다 다르다.

아래 명령어는 전부 저장소 루트에서 실행하고, 경로도 전부 그 기준이다. 저장소 밖에 있는 것은
둘뿐이다 — 모델 `../model/`, lm-eval 하네스 `../dLLM_f/`. 각각 `FUTURE_DLLM_MODEL` /
`FUTURE_DLLM_HARNESS` 환경변수나 `--model` 인자로 바꿀 수 있다.

lm-eval 모델은 한 번만 등록한다. 패키지는 이 저장소에 그대로 두므로 사본이 하나뿐이다:

```bash
ln -sf $(pwd)/eval/LLaDA_future.py ../dLLM_f/eval_model/LLaDA_future.py
echo "from .LLaDA_future import LLaDAFuture" >> ../dLLM_f/eval_model/__init__.py
```

## 데이터셋

`--dataset`으로 프롬프트 소스를 고른다. 생성 길이는 데이터셋에 딸린 값이다 — 평가가 쓰는 값과
같아야 하는데, 이 값이 캐시에서 프롬프트와 모델 출력의 비율을 결정하기 때문이다.

| `--dataset` | 소스 (test 아닌 split) | `--gen-length` | 평가 task |
|---|---|---|---|
| `samsum` | LongBench samsum 대화 | 128 | `longbench_samsum` |
| `gsm8k` | gsm8k train | 128 | `gsm8k`, 5-shot |
| `mmlu` | mmlu validation + dev | 64 | `mmlu_generative` |
| `mbpp` | mbpp validation + prompt | 128 | `mbpp` |
| `math` | hendrycks_math train | 256 | `minerva_math` |
| `mbpp_full` | mbpp full train | 256 | `mbpp` |
| `samsum_lb` / `trec_lb` / `wiki2_lb` | LongBench 형식, ~2048토큰 프롬프트 | 128 / 96 / 64 | `longbench_*` |

추가하려면 `teacher/build_prompt_shards.py`에 builder를 쓰고 `BUILDERS`에 등록한다.

## 1. 교사 라벨 추출

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py     --dataset samsum --n-samples 300 --gen-length 128
```

`artifacts/prompt_shards/<dataset>/`와 `artifacts/teacher/<dataset>/`에 저장된다. 둘 다 이미 있는
샘플은 건너뛰므로, 중단된 실행은 이어받고 개수를 올리면 없는 것만 추가로 추출한다.

## 2. scorer 학습

```bash
python student/train_student.py --teacher-root artifacts/teacher/samsum
```

교사 루트를 쉼표로 여러 개 주면 도메인 혼합 학습이 된다. 체크포인트는 `artifacts/ckpts/<이름>`에
저장되고, 이름에는 구분에 필요한 것만 들어간다 — `1ds_300_e6_lr2e-4_6a5fc6`은 도메인 1개,
300샘플, 6 epoch, lr 2e-4, 그리고 도메인 이름들의 해시다. `--epochs`, `--lr`, `--name`을 바꿀 수
있고, 사용한 설정 전체는 옆의 `meta.json`에 기록된다.

## 3. 추론

```bash
scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
scripts/run_eval.sh gsm8k  1.0        # 축출 없음, scorer 불필요
```

인자는 데이터셋, `keep_ratio`, 체크포인트다. 실행마다 json 하나가
`results/<데이터셋>_keep<비율>_<날짜시간>.json`으로 남는다. `keep_ratio`가 1.0 미만이면
체크포인트가 필수이고, `1.0`이면 축출을 끈다. 문항 수는 `LIMIT`으로 바꾼다 (기본 200).

생성 없이 held-out 라벨로 scorer만 확인하려면:

```bash
python student/eval_recall.py --student artifacts/ckpts/<이름>/checkpoint-best
```
