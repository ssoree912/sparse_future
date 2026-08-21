# tasks

```
metrics.py            LongBench's scoring, unchanged
generate_tasks.py     regenerates longbench/ from LongBench's own config
longbench/            one yaml per task, nothing else
```

The yaml files are generated from LongBench's `config/dataset2prompt.json` and
`config/dataset2maxlen.json`, against the official dataset:

    https://huggingface.co/datasets/zai-org/LongBench

Data is expected at `../data/longbench/data/<task>.jsonl` — official schema,
`input` plus a raw `context`. The prompt lives in `doc_to_text`, which is where
LongBench puts it; a repackaged copy that bakes the prompt into `context` will
render it twice.

`scripts/run_eval.sh` copies `metrics.py` and the yaml files into one scratch
directory at run time, substituting the data path, because lm-eval resolves
`!function metrics.…` next to the yaml.
