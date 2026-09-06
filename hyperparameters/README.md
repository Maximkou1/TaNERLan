# Hyperparameter experiments

Separate from `augmentation/` (train-set variants) and `cleaning/` (train-set
filtering) — this tracks experiments that keep the train set fixed
(`data/train.jsonl`, unmodified) and vary `baseline/train.py`'s
hyperparameters instead. No new code: every run here is a plain
`baseline.train` invocation with one flag changed from its default.

## Seed variance — establishing the noise floor

Before reading any weight given to a single-run delta elsewhere in this
project (`augmentation/README.md`, `cleaning/README.md`): every comparison
so far is **one run per config**. Two runs of the identical config, differing
only in `--seed`, will not produce identical dev metrics — data shuffling
order and weight initialization both depend on it. This experiment measures
how much they differ, so a delta smaller than that can't be trusted as a
real effect from a single run.

```
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/baseline (seed 42, already existed)
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/baseline_seed1 --seed 1
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/baseline_seed7 --seed 7
```

Everything else (3 epochs, batch size 8, lr 5e-5, ...) left at default — the
only variable across these three runs is `--seed`.

| seed | ORG f1 | NAME f1 | GEO f1 | micro P | micro R | micro F1 | macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 (`artifacts/baseline`) | 0.6478 | 0.6062 | 0.7015 | 0.7627 | 0.5744 | 0.6553 | 0.6518 |
| 1 | 0.6496 | 0.6074 | 0.7084 | 0.7766 | 0.5718 | 0.6587 | 0.6552 |
| 7 | 0.6513 | 0.6128 | 0.7081 | 0.7741 | 0.5770 | 0.6612 | 0.6574 |
| **range (max-min)** | 0.0035 | 0.0066 | 0.0069 | 0.0139 | 0.0052 | **0.0059** | 0.0056 |

**Noise floor: roughly ±0.003-0.007 f1** depending on scope (micro-F1 spans
0.6553-0.6612 across just 3 seeds; per-label f1 is noisier still, up to
0.0069 on GEO). Any single-run delta smaller than this can't be told apart
from seed noise without repeating it across seeds.

### Re-reading the other experiments' deltas against this floor

| experiment | micro-F1 delta vs. baseline (single run) | vs. ~0.006 noise floor | verdict |
|---|---:|---|---|
| `augmentation` step 1, +normalization only | -0.0011 | well inside | **not distinguishable from baseline** |
| `augmentation` step 1, +transliteration | +0.0165 | ~2.8x | likely real |
| `augmentation` step 2, +external corpora (pooled) | +0.0108 | ~1.8x | likely real |
| `augmentation` step 2, cascaded (vs. baseline) | +0.0017 | well inside | **not distinguishable from baseline** — walks back the earlier "cascade barely clears baseline" framing to "statistically flat", not a small real gain |
| `augmentation` step 2, cascaded vs. pooled | -0.0090 | ~1.5x | likely real, closer to the boundary than translit/external |
| `augmentation` step 3, +synthetic typos | -0.0108 | ~1.8x | likely real (regression) |
| `cleaning`, dirty-data removal | -0.0020 | well inside | **not distinguishable from baseline** |

Three results that were already described as "flat" or "essentially a wash"
(normalization, cascaded-vs-baseline, cleaning) turn out to be exactly that
in a stricter sense too — their deltas sit inside the range three identical
baseline reruns produce from seed alone, so "no real effect" is the accurate
read, not just "a small one." The larger deltas (transliteration, external
corpora pooled, synthetic augmentation, cascaded-vs-pooled) all clear the
noise floor by 1.5-3x, so those stay credible as real effects on a single
run, though a proper confidence interval would need repeats of those too,
not just of baseline.

### Practical takeaway for further experiments

Don't reduce `--epochs` below 3 as a way to speed up future runs (raised
earlier, corrected in conversation) — the "epoch 3 is worse than epoch 2"
pattern was only ever observed at the current learning rate / weight decay,
and both directly affect how many epochs are needed to converge. Changing
either without keeping epoch count fixed risks confounding the hyperparameter
sweep with an early cutoff.

## Learning rate

All three runs keep `data/train.jsonl`, seed 42, 3 epochs, batch size 8,
weight decay 0.01 — only `--learning-rate` changes.

```
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_lr_2e5 --learning-rate 2e-5
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_lr_3e5 --learning-rate 3e-5
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_lr_1e4 --learning-rate 1e-4
```

| lr | ORG f1 | NAME f1 | GEO f1 | micro F1 | delta vs. baseline (5e-5) |
|---|---:|---:|---:|---:|---:|
| 2e-5 | 0.6183 | 0.5964 | 0.6754 | 0.6326 | **-0.0227** |
| 3e-5 | 0.6301 | 0.5996 | 0.6927 | 0.6437 | -0.0116 |
| 5e-5 (`artifacts/baseline`) | 0.6478 | 0.6062 | 0.7015 | 0.6553 | — |
| 1e-4 | 0.6409 | 0.6064 | 0.6961 | 0.6506 | -0.0047 (inside noise floor) |

Monotonic: lower lr is strictly worse, and the gap to baseline shrinks as lr
climbs toward and past the default. Every run still shows `best_dev_loss` at
epoch 2 (so the "epoch 3 worse" pattern does hold across this lr range —
confirmed, not assumed), meaning this is underfitting within a fixed epoch
budget, not overfitting: a lower lr needs more than 3 epochs to reach where
5e-5 already is by epoch 2, exactly the risk flagged above for not touching
epoch count independently. 1e-4 comes closest (delta inside the ~0.006 noise
floor from the seed-variance section) but still doesn't beat 5e-5. **5e-5
stays the best of the four tested; nothing in [2e-5, 1e-4] improves on it at
3 epochs.**

## Weight decay

Same fixed variables, only `--weight-decay` changes (default 0.01).

```
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_wd_0 --weight-decay 0.0
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_wd_005 --weight-decay 0.05
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_wd_01 --weight-decay 0.1
```

| weight decay | ORG f1 | NAME f1 | GEO f1 | micro F1 | delta vs. baseline (0.01) |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.6475 | 0.6066 | 0.7008 | 0.6551 | -0.0002 |
| 0.01 (`artifacts/baseline`) | 0.6478 | 0.6062 | 0.7015 | 0.6553 | — |
| 0.05 | 0.6469 | 0.6057 | 0.7032 | 0.6554 | +0.0001 |
| 0.1 | 0.6454 | 0.6063 | 0.7041 | 0.6554 | +0.0001 |

All four points sit within 0.0003 micro-F1 of each other — an order of
magnitude below the ~0.006 noise floor. Weight decay in [0.0, 0.1] has no
measurable effect on this dataset/epoch budget; not worth further tuning
here.

## Batch size

Same fixed variables (lr 5e-5, weight decay 0.01), only `--batch-size`
changes (default 8); `--gradient-accumulation-steps` left at 1, so this also
changes the number of optimizer steps per epoch, not just the tensor shape.

```
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_bs_4 --batch-size 4
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_bs_16 --batch-size 16
python -m baseline.train --train data/train.jsonl --dev data/dev.jsonl --output-dir artifacts/hp_bs_32 --batch-size 32
```

| batch size | ORG f1 | NAME f1 | GEO f1 | micro F1 | delta vs. baseline (8) |
|---|---:|---:|---:|---:|---:|
| 4 | 0.6463 | 0.6126 | 0.7097 | 0.6593 | +0.0040 (inside noise floor) |
| 8 (`artifacts/baseline`) | 0.6478 | 0.6062 | 0.7015 | 0.6553 | — |
| 16 | 0.6358 | 0.6101 | 0.7082 | 0.6542 | -0.0011 (inside noise floor) |
| 32 | 0.6209 | 0.5971 | 0.6890 | 0.6385 | **-0.0168** |

4 and 16 are indistinguishable from baseline; 32 drops clearly outside the
noise floor. Same mechanism as the learning-rate sweep: batch size 32 at a
fixed lr of 5e-5 means 4x fewer optimizer steps per epoch than batch size 8,
so it's the same underfitting-within-a-fixed-epoch-budget effect as a too-low
lr, not a batch-size effect per se — lr would need to scale up with batch
size to compensate, not tested here. Batch size 8 (default) stays at or
better than everything tested.

## Summary across all three sweeps

Nothing tested beat the default configuration (lr 5e-5, weight decay 0.01,
batch size 8, 3 epochs) — every deviation either sat inside the seed noise
floor or was a clear regression traceable to underfitting within the fixed
3-epoch budget (too-low lr, or too-large batch size without a compensating
lr increase). The default is already close to a local optimum for this
task/dataset size; further gains are more likely to come from the
augmentation/cleaning experiments than from hyperparameters in these ranges.
