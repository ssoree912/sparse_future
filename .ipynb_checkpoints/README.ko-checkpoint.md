# sparse_future — 실행 안내

채점 기준과 재선택을 수식으로 정리한 문서는 [METHOD.ko.md](METHOD.ko.md)에 있습니다.

LLaDA-8B-Instruct 위에서 [Sparse-dLLM](https://github.com/OpenMOSS/Sparse-dLLM)의 KV 캐시
eviction을 다시 살펴본 실험 모음입니다. 핵심 질문은 하나입니다 — **캐시를 10%로 줄였을 때
잃은 성능을, 토큰을 더 잘 고르는 것으로 얼마나 되찾을 수 있는가.**

결론부터 말하면 **더 잘 고르는 것보다 다시 고르는 것이 훨씬 큽니다.** 그리고 다시 고르려면
버린 토큰을 되살릴 수 있어야 하므로 메모리를 내주게 되는데, 그 메모리는 **K만 남기고
양자화하면 대부분 회수됩니다.**

## 핵심 결과

SAMSum (LongBench, 2048 컨텍스트, 200샘플, ROUGE-L):

| 구성 | 점수 | GPU 캐시 |
|---|---|---|
| eviction 없음 (`keep=1.0`) | 40.02 | 1,124 MB |
| **baseline** (원본 Sparse-dLLM, keep 0.1) | **33.89** | **112 MB** |
| **우리 기준만** (재선택 없음) | **36.19** | **112 MB** |
| 우리 기준 + 재선택 8스텝 | 35.85 | 1,236 MB |
| + V 오프로딩 | 35.85 | 674 MB |
| **+ int4 K 양자화** | **36.48** | **257 MB** |

MATH-500 (100문제, 생성 256토큰, 정확도 — **math_verify 표준 채점**):

| keep_ratio | baseline | 우리 기준만 | 재선택+int4 |
|---|---|---|---|
| 1.0 | 32 | — | — |
| 0.5 | 31 | — | — |
| 0.3 | 8 | — | — |
| **0.25** | **6** | **11** | **18** |
| 0.1 | 2 | — | — |

> 자체 채점기로는 각각 27 / 27 / 7 / 3·8·14 / 1이었습니다. `\frac{1}{2}` vs `0.5` 같은
> 동치 표현을 놓쳐 **일괄적으로 짜게** 나왔고, 특히 답변 기반 라벨이 +5~6씩 손해를 봤습니다.
> MATH 수치는 반드시 `scripts/rescore_math.py`로 재채점해서 쓰세요.

라벨 정의 비교 (SAMSum keep 0.1, 200샘플 / MATH keep 0.25, 100샘플):

| 선택 근거 | SAMSum | MATH | 배포 가능 |
|---|---|---|---|
| baseline (원본 기준) | 33.89 | 6 | ✅ |
| `answer` (확정 직후 1회) | 34.24 | 9 | ❌ 오라클 |
| `final` (완성된 답변 1회) | 34.72 | 8 | ❌ 오라클 |
| `confirmed` (확정 행 매 스텝) | 35.27 | 7 | ❌ 오라클 |
| **`ours` (마스크 행)** | **36.19** | **11** | **✅** |

미래를 아는 오라클들이 현재만 보는 방법을 못 이깁니다. 예산은 여섯 모드 모두 레이어당
정확히 `int(후보수 × keep_ratio)`개로 **실측 확인**했습니다 (`scripts/check_budget.py`).

## 무엇이 eviction이고 무엇이 아닌가

**채점 기준 변경은 eviction이 맞고, 메모리를 전혀 더 쓰지 않습니다.** 두 기준 모두 스텝 1,
즉 전체 forward가 모든 위치의 K를 막 만들어낸 시점에 돌아갑니다. baseline도 그 K를 손에
쥐고 있고, 둘 다 고른 직후 나머지를 버립니다. 같은 112 MB, 같은 한 번의 결정, 보관 없음,
학습 없음인데 **33.89 → 36.19**입니다.

**재선택은 eviction이 아닙니다.** 버린 항목을 되살리려면 그게 남아 있어야 하므로 풀을
보관하게 됩니다(압축·오프로딩하더라도). baseline 대비 GPU 캐시 2.3배에 호스트 메모리와
PCIe 전송이 추가되고, eviction을 안 한 경우 대비로만 4.4배 작습니다. 캐시 eviction이라기보다
**상주 색인을 둔 paged sparse attention**이라고 부르는 게 정확합니다. SAMSum에서는 기준
변경 위에 +0.29만 더합니다.

## 두 가지 scorer

`CustomCache`가 두 채점 기준을 **나란히** 들고 있고, `scorer` 값으로 고릅니다.
**기본값은 원본**이므로 명시하지 않으면 우리 방법이 켜지지 않습니다.

| `scorer=` | 순위 계산 |
|---|---|
| `'sparse_dllm'` (기본) | 블록의 쿼리 32개를 **평균**해서 벡터 하나로 만든 뒤 K와 원시 내적 → 헤드 평균 → max-pool |
| `'masked_row'` (우리) | 쿼리 **행별로** 점수 계산 → **softmax** → 헤드 평균 → **아직 [MASK]인 행만 합산** → max-pool |

세 번째 차이가 결정적입니다. 블록 중간에는 32개 행 중 상당수가 이미 확정되어 캐시가 더
필요 없는데, 쿼리를 평균내면 그 확정된 토큰들이 순위를 지배합니다. 같은 재선택 주기에서
원본 공식을 쓰면 **33.65**로 재선택을 안 한 것(33.89)보다 나쁘고, `masked_row`를 쓰면
**35.85**가 됩니다.

`scorer`와 `reselect_every`는 **독립**입니다. 기준만 바꾼 경우와 주기까지 바꾼 경우를
따로 볼 수 있습니다.

## 설정 인자

| 인자 | 뜻 |
|---|---|
| `keep_ratio` | 후보 중 남길 비율 (0.1 = 10%) |
| `scorer` | `'sparse_dllm'` 또는 `'masked_row'` |
| `reselect_every=R` | 스텝 1의 K/V를 보관해두고 R스텝마다 다시 순위를 매김. **재계산은 없음**, 남길 창만 이동 |
| `reselect_offload_v=True` | V를 CPU에 두고 선택된 것만 GPU로 전송. **출력이 완전히 동일** (5/5 검증) |
| `reselect_k_bits=8` 또는 `4` | 보관하는 K를 양자화. K는 순위만 정하므로 손실이 없음 (int4가 오히려 더 높음) |
| `oracle_eviction=True` | 연구용. 블록의 남은 스텝 어텐션을 미리 보고 고름 (배포 불가) |

## 설치

```bash
# 1) 모델 코드 교체
cp sparse_dllm/*.py <opencompass>/opencompass/models/sparse_dllm/

# 2) 설정 배치
cp configs/sparse_llada_*.py            <opencompass>/myeval/models/
cp configs/longbench_passage_*.py       <opencompass>/myeval/datasets/

# 3) site-packages에 별도 설치본이 있으면 거기에도 복사 (아래 주의사항 참고)
python -c "import opencompass, os; print(os.path.dirname(opencompass.__file__))"
cp sparse_dllm/*.py <위에서 나온 경로>/models/sparse_dllm/
```

## 실행 명령어

### SAMSum (LongBench)

```bash
cd <opencompass>
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=<hf_cache> HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# baseline (원본 Sparse-dLLM, keep 0.1)
python run.py --config-dir myeval --models sparse_llada_longbench_keep01 \
  --datasets longbench_samsum_gen --work-dir outputs/samsum_baseline --max-workers-per-gpu 1

# 우리 기준만 (재선택 없음) — 기준의 효과만 분리
python run.py --config-dir myeval --models sparse_llada_ours_oneshot_keep01 \
  --datasets longbench_samsum_gen --work-dir outputs/samsum_ours --max-workers-per-gpu 1

# 우리 기준 + 재선택 8스텝
python run.py --config-dir myeval --models sparse_llada_reselect8_keep01 \
  --datasets longbench_samsum_gen --work-dir outputs/samsum_reselect8 --max-workers-per-gpu 1

# 권장 구성: 우리 기준 + 재선택 + V 오프로딩 + int4 K
python run.py --config-dir myeval --models sparse_llada_reselect8_int4k \
  --datasets longbench_samsum_gen --work-dir outputs/samsum_int4k --max-workers-per-gpu 1

# eviction 없음 (천장)
python run.py --config-dir myeval --models sparse_llada_keep10 \
  --datasets longbench_samsum_gen --work-dir outputs/samsum_keep10 --max-workers-per-gpu 1
```

점수는 `<work-dir>/<타임스탬프>/results/<abbr>/LongBench_samsum.json`에 나옵니다.

### passage_retrieval (4096 컨텍스트)

```bash
python run.py --config-dir myeval --models sparse_llada_reselect8_int4k \
  --datasets longbench_passage_retrieval_short_gen \
  --work-dir outputs/pr_int4k --max-workers-per-gpu 1
```

`longbench_passage_retrieval_short_gen`은 LLaDA의 4096 한계 안에 온전히 들어가는 78개만
추린 것입니다. 원본은 컨텍스트 중앙값이 9,255토큰이라 정답 문단이 잘려나가서, eviction
때문에 틀린 것인지 잘림 때문인지 구분되지 않습니다.

### MATH-500

OpenCompass를 거치지 않는 전용 러너입니다. 모델을 한 번만 올리고 여러 구성을 순차 실행합니다.

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=2 python scripts/run_math500_sparse.py \
  --n-samples 100 \
  --variants keep1.0 keep0.25 keep0.25-reselect8-int4k
```

변형 이름은 문자열로 조합합니다: `keep<비율>`, `-reselect<주기>`, `-int<비트>k`,
`-offloadv`, `-oracle`, `-oracle-perstep-every<주기>`. 결과는
`results/math500/<변형>.json`에 정확도·소요시간·peak 메모리와 함께 저장됩니다.

### lm-eval (공식 하네스)

MATH·GSM8K는 OpenCompass가 아니라 **lm-evaluation-harness**가 표준입니다. `dLLM_f`의
lm-eval 경로에 Sparse-dLLM eviction을 붙인 래퍼가 `eval_model/LLaDA_sparse.py`이고,
모델 코드는 OpenCompass 의존성 없이 쓰도록 `dLLM_f/sparse_dllm/`에 독립 복사해 둡니다.

```bash
cd <dLLM_f>
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export HF_HOME=<hf_cache> HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python evaluation_script.py \
  --model LLaDA_sparse --tasks minerva_math --batch_size 1 --limit 200 \
  --model_args "pretrained=<model>,keep_ratio=0.25,scorer=masked_row,block_len=32" \
  --gen_kwargs "block_length=32,gen_length=256,steps=256,cfg_scale=0.0" \
  --num_fewshot 0 --apply_chat_template --fewshot_as_multiturn
```

`scorer=sparse_dllm`이면 baseline입니다. `--model_args`로 `reselect_every`,
`reselect_k_bits`, `oracle_*`도 전부 넘길 수 있고, 로딩 시 `[LLaDA_sparse] ...` 로그로
실제 전달된 값이 찍힙니다.

주의: dLLM_f의 기존 `.sh`는 `-m lm_eval` 인자를 붙이는데 lm_eval 0.4.12에서는
`unrecognized arguments` 오류가 납니다 — 빼고 실행하세요. 그리고 그 스크립트들은
`block_length=256`(블록 1개)이라 우리 방법의 전제와 다릅니다. **`block_length=32`로
맞춰야** 지금까지의 실험과 비교됩니다.

### 메모리 측정

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/bench_cache_memory.py --task samsum --n-samples 5
CUDA_VISIBLE_DEVICES=2 python scripts/bench_cache_memory.py --task passage_retrieval --n-samples 5
```

구성별로 **캐시가 실제로 점유한 GPU 바이트**, 전체 peak, 샘플당 시간, 그리고 첫 번째
구성 대비 출력이 같은지를 출력합니다. `--variants`로 비교 대상을 고를 수 있습니다.

## 주의사항

**site-packages 문제 (가장 중요).** 환경에 `opencompass`가 editable이 아닌 형태로 설치돼
있으면, OpenCompass가 띄우는 **추론 서브프로세스가 그 설치본을 import**합니다. 작업
트리만 고치면 새 인자가 조용히 무시되고 로그에 `WARNING - Unused argument <이름>=True`
한 줄만 남은 채 **예전 코드가 그대로 돌아갑니다.** 이 세션에서 이것 때문에 200샘플 실행
하나가 통째로 헛돌았습니다. 코드 변경이 실제로 적용됐는지는 점수가 아니라 **로그의 디버그
출력이나 예측 변화**로 확인하세요 — 스모크 점수는 예전 경로와 우연히 같을 수 있습니다.

**간헐적 segfault.** student/oracle 경로에서 `libcuda.so` 내부의 같은 오프셋에서
무작위로 segfault가 납니다 (NVRM Xid 없음, 데이터 의존성 없음). `scripts/run_samsum_with_retry.sh`
가 `-r <타임스탬프>`로 이어받아 재시도합니다 — OpenCompass가 부분 예측에서 재개하므로
손실이 없습니다.

**GPU는 하나씩.** 모델이 bf16으로 약 16.7 GB를 쓰므로 24 GB 카드에서 두 실행을 동시에
띄우면 OOM입니다. 순차로 돌리세요.

**MATH 채점.** `scripts/run_math500_sparse.py`의 답 매칭은 직접 짠 것이라 동치 표현을
놓칩니다. 결과 json이 예측 텍스트를 그대로 담고 있으므로 `scripts/rescore_math.py`로
`math_verify` 재채점을 돌리세요 (GPU 불필요, 수 초). 방법론 간 비교는 반드시 재채점 후에
하십시오 — 라벨에 따라 손해 폭이 달라 순서가 바뀝니다.

**속도.** 재선택 자체는 스텝 하나의 3.5%(8스텝 주기면 평균 0.44%)라 실측 +4%입니다.
V 오프로딩을 켜면 CPU→GPU 전송 때문에 +11%가 됩니다. pinned memory를 쓰면 더 줄일 수
있지만 아직 적용하지 않았습니다.
