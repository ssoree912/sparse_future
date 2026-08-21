# future_dllm

디퓨전 LLM의 KV 캐시 축출을, 지금 블록이 보고 있는 것이 아니라 **완성된 답변이 필요로 할 것**을 기준으로 결정한다.

디퓨전 LLM은 한 블록을 여러 디노이징 스텝에 걸쳐 만든다. Sparse-dLLM은 블록마다 한 번 캐시를 잘라내는데, 그 순위를 **자르는 시점의 어텐션**으로 매긴다 — 블록의 모든 토큰이 아직 `[MASK]`인 시점이다. 블록이 무슨 말을 할지 정해지기 전에 무엇을 남길지 정하는 셈이다.

`future_dllm`은 완성된 답변이 매겼을 순위를 학습한다. 교사 패스가 전체 캐시로 블록을 끝까지 채우고, 완성된 블록에 forward를 한 번 더 돌려 그 답변 토큰들이 실제로 무엇을 봤는지 기록한다. 작은 scorer가 이것을 자르는 시점에 얻을 수 있는 상태로부터 예측하도록 학습되고, 배포 시에는 블록당 한 번의 선택만 한다 — 추가 forward도, 추가 메모리도 없다.

## 라벨

블록의 행 `r`(완성된 답변 토큰)과 캐시 후보 `j`에 대해:

```
a_rj = softmax_j ( q_r · k_j / sqrt(d) )          헤드 평균 어텐션
I_j  = max_r a_rj                                  후보별 중요도
```

합이 아니라 **최댓값**이다. 어떤 완성된 토큰 **하나라도** 그 캐시 항목에 강하게 의존했다면 남겨야 하는데, 합을 쓰면 그 토큰이 신경 쓰지 않은 나머지 31개에 묻힌다. SAMSum keep 0.1에서 합 형태는 34.72, 최댓값 형태는 36.63이다. 그래서 이 저장소에는 최댓값 형태만 남아 있다.

scorer는 레이어별로 `I_j` 순위를 맞추도록 학습한다. 정규화된 라벨에 대한 listwise KL과 pairwise 순서 항을 함께 쓰고, 체크포인트는 k ∈ {5, 10, 20, 30, 50}% 평균 recall로 고른다 — 고정된 예산이 아니라 scorer를 얻기 위해서다.

## 결과

`LLaDA-8B-Instruct`, keep ratio 0.1, block length 32. 두 행의 예산이 같은 것을 직접 계측했다 — 블록당 95/100/96개로 동일하고, 다른 것은 고르는 기준뿐이다.

| | SAMSum ROUGE-L¹ | GSM8K flex² | GSM8K strict² |
|---|---|---|---|
| 축출 없음 (keep 1.0) | 40.02 | 0.765 | 0.480 |
| Sparse-dLLM 베이스라인 | 33.89 | 0.455 | 0.140 |
| **future_dllm** | **35.86** | **0.745** | **0.470** |

¹ OpenCompass LongBench, 하네스를 통일하기 전에 측정. ² lm-eval, 5-shot, 200문항.

SAMSum 열은 lm-eval로 옮기기 전 수치이고, 이전 실행들과의 연속성을 위해 남겨둔다. 같은 scorer를 lm-eval `longbench_samsum`으로 재면 31.01이다. 두 하네스가 긴 프롬프트를 반대쪽에서 자르기 때문에(OpenCompass는 앞 2048토큰, lm-eval은 뒤 2048토큰) 두 SAMSum 수치는 서로 비교할 수 없다. 베이스라인과 상한 행은 아직 lm-eval로 재측정하지 않았다.

GSM8K는 축출로 잃은 것의 93%를 되찾고, SAMSum은 32%다. 이 차이는 캐시에 무엇이 들어 있는지를 따라간다 — 수학 블록의 캐시는 모델 자신의 추론 사슬이라 중간 결과 하나를 잃으면 그 뒤가 전부 무너지지만, 요약 블록의 캐시는 입력 텍스트라 어떤 순위를 써도 10%로는 복원되지 않는다.

하나의 scorer가 네 도메인을 모두 덮는다. 요약만으로 학습하면 도메인 안에서 recall 0.752, 밖에서 0.625~0.692다. 혼합으로 학습하면 도메인 안 0.752를 유지하면서 밖에서 0.760~0.771이 된다. 약점은 표현력이 아니라 데이터 부족이었다.

## 환경

```bash
conda activate dllm          # torch 2.5.1+cu124, transformers 4.46.3, lm-eval 0.4.12
```

모델: `LLaDA-8B-Instruct`, 경로 `/workspace/dllm/model/LLaDA-8B-Instruct`. GPU 1장, batch size 1 — 캐시 상태가 시퀀스마다 다르기 때문이다.

lm-eval 모델은 한 번만 설치한다. 패키지는 이 저장소에 그대로 두므로 관리할 사본이 하나뿐이다:

```bash
HARNESS=/workspace/dllm/dLLM_f
ln -sf $(pwd)/eval/LLaDA_future.py $HARNESS/eval_model/LLaDA_future.py
echo "from .LLaDA_future import LLaDAFuture" >> $HARNESS/eval_model/__init__.py
```

## 실행

세 단계다. 나머지는 고정이다 — block length 32, greedy 디코딩, 블록당 한 번 선택.

**1. 프롬프트 → 교사 라벨.** 프롬프트 shard는 test가 아닌 split에서만 만든다. 교사가 평가 항목을 보는 일이 없다.

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py --dataset samsum --n-samples 300 --gen-length 128
```

**2. scorer 학습.** 교사 루트를 쉼표로 여러 개 주면 도메인 혼합 학습이 된다. val은 도메인별로 나누고 체크포인트는 도메인 macro 평균으로 고른다.

```bash
python student/train_student.py \
  --teacher-root results/budget/teacher/samsum \
  --output-dir  results/budget/student/samsum
```

**3. 평가.** 공식 하네스는 lm-eval 하나다.

```bash
cd $HARNESS
python evaluation_script.py --model LLaDA_future \
  --model_args "pretrained=$MODEL,keep_ratio=0.1,student_path=$CKPT" \
  --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size 1 \
  --gen_kwargs "block_length=32,gen_length=256,steps=256,temperature=0.0"
```

생성 평가를 돌리기 전에 held-out 라벨로 scorer를 값싸게 확인할 수 있다:

```bash
python student/eval_recall.py --student results/budget/student/samsum/checkpoint-best
```

## 옵션

바꾸도록 만들어둔 것은 두 개다.

| 옵션 | 위치 | 의미 |
|---|---|---|
| `keep_ratio` | `--model_args` | 블록마다 남기는 캐시 후보의 비율. `1.0`이면 축출을 끄고 scorer가 필요 없다. `1.0` 미만이면 `student_path`가 필수다 |
| `--dataset` | `build_prompt_shards.py`, `extract_teacher.py` | 어떤 프롬프트 소스로 라벨을 만들지 |

생성 길이는 사용자 옵션이 아니라 데이터셋에 딸린 값이다. 평가가 쓰는 값과 같아야 하는데, 이 값이 캐시에서 프롬프트와 모델 출력의 비율을 결정하기 때문이다:

| dataset | 프롬프트 | `--gen-length` | 평가 task | 평가 `gen_length` |
|---|---|---|---|---|
| `samsum` | 대화 | 128 | `longbench_samsum` | 128 |
| `gsm8k` | train split | 128 | `gsm8k` (5-shot) | 256 |
| `mmlu` | validation + dev | 64 | `mmlu_generative` | 64 |
| `mbpp` | validation + prompt | 128 | `mbpp` | 256 |
| `math` | `hendrycks_math` train | 256 | `minerva_math` | 256 |

`samsum_lb`, `trec_lb`, `wiki2_lb`는 LongBench 형식의 긴 프롬프트(~2048 토큰)를 만든다. 위 결과에는 쓰이지 않았다.

## 구성

```
future_dllm/     모델: CustomCache(축출), generate(블록 단위 디코딩), scorer 모듈
teacher/         프롬프트 shard와 final × row-max 라벨 추출기
student/         scorer 학습, held-out 라벨 recall 확인
eval/            lm-eval 모델, LLaDA_future로 등록
```

## 메모

- scorer는 저장된 은닉 상태가 아니라 `x_at_block_start`를 학습 시점에 재생해서 본다. 그래서 특징이 배포와 정확히 일치한다. 이전 버전은 프롬프트만 있는 forward로 학습하고 프롬프트+생성 forward로 배포했는데, 이 재생이 그 불일치를 없앤다.
- 프롬프트와 suffix가 하나의 top-k에서 경쟁하므로 예산이 정확히 `후보 수 × keep_ratio`이고, 베이스라인의 계산과 일치한다.
- lm-eval에 `--use_cache <dir>`을 주면 재개가 된다. 긴 작업에서는 이게 중요하다.
