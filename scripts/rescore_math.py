"""Re-score the saved MATH predictions with math_verify (the standard checker).

Nothing is regenerated — this only replaces my hand-written answer matching, so
any change is purely a scoring-harness effect.
"""
import json, glob, os
from math_verify import parse, verify

R = '/workspace/dllm/Sparse-dLLM/results/math500/'
order = ['keep1.0','keep0.5','keep0.3','keep0.25','keep0.25-ours','keep0.25-oracle',
         'keep0.25-oracle-answer','keep0.25-oracle-final','keep0.25-oracle-confirmed',
         'keep0.25-reselect8-int4k','keep0.1']
print(f'{"variant":32s} {"내 채점":>8s} {"math_verify":>12s} {"차이":>6s}')
for v in order:
    p = R + v + '.json'
    if not os.path.exists(p):
        continue
    recs = json.load(open(p))['records']
    mine = sum(r['correct'] for r in recs)
    hit = 0
    for r in recs:
        try:
            gold = parse('$' + r['gold'] + '$')
            pred = parse(r['prediction'])
            if verify(gold, pred):
                hit += 1
        except Exception:
            pass
    print(f'{v:32s} {mine:8d} {hit:12d} {hit-mine:+6d}')
