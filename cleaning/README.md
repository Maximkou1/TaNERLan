# Data-cleaning experiment

A separate, one-off experiment on the plain baseline train set — not part of
the `augmentation/` ladder (which only ever *adds* records to
`data/train.jsonl`). Question: does dropping noisy-formatting records from
train help the model, or hurt it by losing gold labels along the way?

## Auditing `data/train.jsonl` for "dirty" data first

Checked before assuming anything was actually wrong:

- **Structural issues are already impossible.** `baseline/common.py`'s loader
  rejects duplicate hashes, invalid label/offsets, duplicate entities, and
  overlapping entities at load time — `data/train.jsonl` already passed this
  or training would never have run on it.
- **Exact-duplicate texts:** 0. **Near-duplicates** (casefolded, whitespace-
  normalized): 1 pair.
- **Lowercase-initial entity mentions:** 6.6% (4,347/66,083) — the same
  heuristic that found real generic-noun mislabeling in the external NER
  corpora (`augmentation/README.md` step 2) mostly finds *legitimate* short
  mentions here on inspection: hashtags (`daryolive`, `adidasfootball`),
  nicknames (`V`, `뷔` — K-pop idols), casual lowercase writing. Not filtered.
- **Single-character entities:** 77, nearly all legitimate (`X` = the
  platform, `V`/`뷔` = K-pop nicknames). Not filtered.
- **Entity-density outliers** (>8 entities per 100 chars): 207, all
  legitimate — short lists of names (`Iverson Kobe Bron Dirk Shaq`). Not
  filtered.

None of the "wrong annotation" patterns that mattered for external corpora
showed up here. What *did* show up is stylistic noise from the dataset's
social-media origin, chosen as the filter criteria after checking counts and
overlap:

| criterion | records | definition |
|---|---:|---|
| ALL-CAPS | 557 | >95% of alphabetic characters uppercase (min 10 letters) — spam/blessing-style posts |
| Stretched letters | 335 | 4+ identical characters in a row (`ЯАШАНГГГГГГГ`, decorative runs like `⊹⊹⊹⊹⊹⊹⊹⊹⊹⊹`) |
| Very short text | 400 | under 20 characters |
| **union (dropped)** | **1,267** | a record matching *any* of the three |

## What it does

Code: [`filter_dirty.py`](filter_dirty.py).

```
python -m cleaning.filter_dirty \
  --input data/train.jsonl \
  --output cleaning/data/train_clean.jsonl \
  --stats cleaning/data/filter_stats.json
```

`data/train.jsonl` (13,000) → `train_clean.jsonl` (11,733). This is a
**removal**, not an addition — record count goes down, and with it 5,557 of
66,083 gold entities (8.4%: 1,188 ORG, 2,753 NAME, 1,616 GEO) that happened
to live inside the dropped records. That's the real cost being weighed here:
some of what gets removed as "noisy formatting" still carries gold labels
the model would otherwise have learned from.

## Training and evaluation

```
python -m baseline.train \
  --train cleaning/data/train_clean.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/clean_baseline

python -m baseline.predict \
  --model-dir artifacts/clean_baseline/model \
  --input data/dev.jsonl \
  --output artifacts/clean_baseline/dev_predictions.jsonl

python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/clean_baseline/dev_predictions.jsonl \
  --output artifacts/clean_baseline/dev_metrics.json
```

Ran on 11,733 train records, 3 epochs, same seed/hyperparameters as
`artifacts/baseline` — the only variable is the (smaller) train set.

## Results

| scope | metric | baseline | cleaned (-1,267 records) | delta |
|---|---|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6368 | **-0.0110** |
| NAME | f1 | 0.6062 | 0.6086 | +0.0024 |
| GEO | f1 | 0.7015 | 0.7044 | +0.0029 |
| micro | precision | 0.7627 | 0.7550 | -0.0077 |
| micro | recall | 0.5744 | 0.5757 | +0.0013 |
| micro | f1 | 0.6553 | 0.6533 | -0.0020 |
| macro | f1 | 0.6518 | 0.6499 | -0.0019 |

Net negative, though small — essentially a wash on NAME/GEO, driven by ORG
(-1.1 f1 points, the single largest move on either side). This matches a
pattern across `augmentation/README.md`'s other experiments too: ORG is
consistently the label most sensitive to changes in what the model sees
during training, in both directions. A plausible reading here: ALL-CAPS/
spam-style posts disproportionately mention organizations (companies,
institutions, brands in promotional or complaint posts), so removing them as
"noise" also removed a disproportionate share of ORG training signal and
surface-form diversity — the lost gold entities cost more than the removed
formatting noise saved. **Not adopted** — `data/train.jsonl` (unfiltered)
stays the actual baseline; this file documents a real negative result, same
spirit as `augmentation/README.md`'s abandoned attempt 1.
