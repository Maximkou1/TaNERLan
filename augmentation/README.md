# Augmentation experiments

Three independent additions to the baseline train set, run and evaluated one
at a time so each one's effect on dev metrics is attributable on its own:

1. **Transliteration** — Cyrillic<->Latin script-flipped duplicates of
   `data/train.jsonl`. Split into two separately-trained arms so the effect
   of normalization alone can't hide inside the transliteration number:
   normalizing apostrophes/diacritics on their own first (flat effect), then
   adding the script-flipped duplicates on top of that normalized base (the
   actual gain).
2. **External corpora** — an annotated external Uzbek NER corpus appended to
   train, no transliteration. (First tried raw unlabeled `tahrirchi`
   text — abandoned, it hurt recall. A second annotated corpus was also
   tried on top of the first — it erased the gain, so it isn't part of the
   recommended config either. Also compared pooled vs. cascaded
   (sequential) fine-tuning on the winning config — pooled wins. All
   documented below.)
3. **Synthetic augmentation** — entity-safe typo noise on train, no
   transliteration, no external corpora.

`data/dev.jsonl` is never modified or augmented in any step — it stays the
original distribution the model is judged against.

## Step 1 — transliteration

Two attempts, run and evaluated separately rather than jointly, so each
effect is attributable on its own: apostrophe/diacritic normalization touches
every record and could plausibly move dev metrics by itself, independent of
whether any script-flipped duplicates get added. Attempt 2 is built **on top
of** attempt 1's normalized file (not on raw `data/train.jsonl`), so its
result isolates exactly the effect of adding the duplicates, with
normalization already controlled for in both arms.

### Attempt 1 — apostrophe/diacritic normalization only

Code: [`normalize_dataset.py`](normalize_dataset.py), reusing
[`tanerlan/modern_bert/tokenizer/tokenization_utils.py`](../tanerlan/modern_bert/tokenizer/tokenization_utils.py)'s
`prepare_input(text, homoglyphs=False)`, which normalizes the mix of
apostrophe-like codepoints in this dataset (`'` `` ` `` `’` `‘` `ʼ` ...) into
exactly two canonical characters by role — U+02BB right after o/O/g/G (part
of the letter: `oʻ` = `ў`, `gʻ` = `ғ`), U+02BC between two other letters (the
glottal stop, `ъ`, e.g. `sanʼat`), and left alone when not preceded by a
letter (an actual quotation mark).

**Do `oʻ`/`gʻ` count as one character or two?** Two, always. Uzbek Latin has
no precomposed codepoint for these letters: `len("oʻ") == 2`, and
`unicodedata.normalize("NFC", "oʻ") == "oʻ"` — NFC does not merge them. A full
scan of `data/train.jsonl` also found zero combining diacritics (Unicode
category Mn) attached to any Cyrillic or Latin letter; the only combining
marks present in the corpus belong to unrelated scripts inside quoted foreign
text (Arabic vowel points, Thai, Devanagari — 345 occurrences total, listed in
`transliteration.py`'s docstring). So "normalize diacritic letters to one
format" turns out to be the same problem as apostrophe normalization above —
there is no separate combining-mark pass to write.

This is length-preserving by construction (each apostrophe-like codepoint
maps to exactly one canonical codepoint, verified with an assertion in
`prepare_input`), so entity `start`/`end` offsets are untouched — no
remapping needed, unlike attempt 2 below.

```
python -m augmentation.normalize_dataset \
  --input data/train.jsonl \
  --output augmentation/data/train_normalized.jsonl \
  --stats augmentation/data/normalization_stats.json
```

`data/train.jsonl` (13,000) → `train_normalized.jsonl` (13,000, same records,
2,789 of them with at least one apostrophe variant actually rewritten — no
new records, record count is unchanged).

```
python -m baseline.train \
  --train augmentation/data/train_normalized.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/aug_normalize

python -m baseline.predict \
  --model-dir artifacts/aug_normalize/model \
  --input data/dev.jsonl \
  --output artifacts/aug_normalize/dev_predictions.jsonl

python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/aug_normalize/dev_predictions.jsonl \
  --output artifacts/aug_normalize/dev_metrics.json
```

**Result:** essentially flat, slightly negative.

| scope | metric | baseline | +normalization only | delta |
|---|---|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6401 | -0.0077 |
| NAME | f1 | 0.6062 | 0.6123 | +0.0061 |
| GEO | f1 | 0.7015 | 0.7011 | -0.0004 |
| micro | precision | 0.7627 | 0.7687 | +0.0060 |
| micro | recall | 0.5744 | 0.5694 | -0.0050 |
| micro | f1 | 0.6553 | 0.6542 | -0.0011 |
| macro | f1 | 0.6518 | 0.6512 | -0.0006 |

Canonicalizing apostrophes on the *same* 13,000 records, with no new data
added, does not move dev metrics beyond noise. This matters for interpreting
attempt 2: whatever gain transliteration shows is attributable to the added
script-flipped duplicates themselves, not to this normalization pass riding
along inside them.

### Attempt 2 — transliteration (on top of the normalized base)

Code: [`transliteration.py`](transliteration.py) (the Cyrillic<->Latin engine
and exact entity-offset remapping), [`build_dataset.py`](build_dataset.py)
(CLI that reads a train file and appends transliterated duplicates).

For every train record written **purely** in one script, adds one new record
with the text transliterated into the other script and all entity spans
remapped to the new offsets. Records that mix both scripts (2,952/13,000 —
mostly Russian code-switching) are left as a single, unmodified copy: running
Uzbek transliteration rules over embedded Russian (or vice versa) would
corrupt whichever language the rule doesn't belong to, with no reliable way
to tell which spans are which language without a language ID model.

`transliterate_record` also runs `prepare_input` internally before
script-flipping (same normalization as attempt 1, needed so the digraph
tables in `transliteration.py` see canonical apostrophes) — but instead of
building this on raw `data/train.jsonl`, it's pointed at attempt 1's already-
normalized file, so the pass-through (untransliterated) records end up
normalized too and both arms share the same normalization treatment:

```
python -m augmentation.build_dataset \
  --input augmentation/data/train_normalized.jsonl \
  --output augmentation/data/train_translit.jsonl \
  --stats augmentation/data/transliteration_stats.json
```

Result on `train_normalized.jsonl` (13,000 records):

| script (by letters present) | records | transliterated duplicate added |
|---|---:|---:|
| Cyrillic-only | 2,207 | 2,207 (→ Latin) |
| Latin-only | 7,840 | 7,840 (→ Cyrillic) |
| mixed | 2,952 | 0 (kept once, unmodified) |
| other (no Cyrillic/Latin letters) | 1 | 0 |
| **train_translit.jsonl total** | | **23,047** |

`skipped_unsafe_digraph_boundary: 0` — no gold entity in this dataset has a
boundary landing inside a merged digraph (see below), so nothing had to be
dropped.

### Entity offset remapping

Every original character is mapped to zero, one or two output characters
(`augmentation/transliteration.py:_build_offsets`), and each entity's
`start`/`end` is looked up through the resulting prefix-sum offset array —
exact by construction, not approximated:

- **Cyrillic → Latin** is always safe: one Cyrillic letter always maps to a
  whole, atomic Latin piece (0–2 chars), so any character boundary in the
  source is automatically also a boundary in the output.
- **Latin → Cyrillic** collapses digraphs (`sh` `ch` `ts` `yo` `yu` `ya` `ye`
  `oʻ` `gʻ`) into a single Cyrillic letter. If a gold entity boundary happens
  to fall *inside* one of these digraphs (e.g. between the `s` and the `h` of
  `sh`), the span can't be represented after conversion — `transliterate_record`
  raises `TransliterationSkipped` and the record is dropped from the
  augmentation (counted, never silently mis-shifted). In practice this is
  effectively never triggered: entity boundaries follow word/morpheme edges
  (see `LABELING_GUIDE.md`), essentially never mid-digraph — confirmed by the
  0 skips above.

### Known limitations (accepted, not fixed)

- Neither direction is a certified linguistic round trip. Cyrillic `е`/`э`
  both map to Latin `e` (matching the real Uzbek Latin alphabet, which merged
  them); the Latin→Cyrillic direction is a greedy longest-match heuristic
  that can occasionally mis-split letters that happen to spell a digraph
  (literal `т`+`с` read back as `ц`). Acceptable for augmentation: the
  requirement is plausible script-flipped text with *exactly* correct entity
  spans, not a certified transliteration standard.
- `classify_script` only detects which alphabet a record uses, not which
  language. A handful of Latin-script records in this dataset are plain
  English (social-media noise, e.g. "ENGINE send protest trucks..."), which
  still gets run through the Uzbek Latin→Cyrillic table and comes out
  garbled ("труцкс"). Entity spans stay correct even there; the token
  content is just noisier. Filtering by language was out of scope for this
  step.
- ~55 rare Cyrillic letters from other languages appear in the corpus in
  small numbers (Kazakh/Karakalpak, Ukrainian, Tajik — e.g. `і` 124x, `ә` 53x,
  `ң` 41x, all <0.05% of letters). They pass through unmapped/unchanged
  rather than being force-mapped to an approximate Uzbek letter.

### Training and evaluation

```
python -m baseline.train \
  --train augmentation/data/train_translit.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/aug_translit

python -m baseline.predict \
  --model-dir artifacts/aug_translit/model \
  --input data/dev.jsonl \
  --output artifacts/aug_translit/dev_predictions.jsonl

python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/aug_translit/dev_predictions.jsonl \
  --output artifacts/aug_translit/dev_metrics.json
```

All other hyperparameters (seed 42, 3 epochs, batch size 8, lr 5e-5, ...) are
left at baseline defaults so the only variable between this run and
`artifacts/baseline` is the train set.

### Results

Ran on 23,047 train records (13,000 normalized originals + 10,047
transliterated duplicates), 3 epochs, same seed/hyperparameters as
`artifacts/baseline`. `best_dev_loss` improved from 0.1097 (epoch 2 of
baseline) to 0.1038 (epoch 2 of this run). Exact-span dev metrics,
`artifacts/baseline/dev_metrics.json` vs. `artifacts/aug_normalize/dev_metrics.json`
vs. `artifacts/aug_translit/dev_metrics.json`:

| scope | metric | baseline | +normalization only | +translit (on top) | delta vs. baseline |
|---|---|---:|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6401 | 0.6531 | +0.0053 |
| NAME | f1 | 0.6062 | 0.6123 | 0.6274 | +0.0212 |
| GEO | f1 | 0.7015 | 0.7011 | 0.7252 | +0.0237 |
| micro | precision | 0.7627 | 0.7687 | 0.7970 | +0.0343 |
| micro | recall | 0.5744 | 0.5694 | 0.5805 | +0.0061 |
| micro | f1 | 0.6553 | 0.6542 | 0.6718 | +0.0165 |
| macro | f1 | 0.6518 | 0.6512 | 0.6686 | +0.0168 |

Every label and both precision and recall improved over baseline, most on
GEO (+2.4 f1 points) — consistent with GEO names (`Toshkent`/`Тошкент`,
`Farg'ona`/`Фарғона`, ...) being exactly the kind of short, script-swappable
tokens transliteration duplicates most directly. Since normalization alone
(previous column) is flat, essentially the entire +0.0165 micro-F1 gain here
is attributable to the added script-flipped duplicates, not to the
apostrophe canonicalization riding along inside them.

## Step 2 — external corpora

Three attempts, all kept here since two of them are real, useful negative
(or non-)results, not wasted work:

1. Raw, unlabeled `tahrirchi` text — measurably hurt dev recall, abandoned.
2. The Kaggle `courpusNER2015` corpus, annotated, added to train — a clean
   win, **the recommended config for this step**.
3. Adding a second annotated corpus (Mendeley Gold) on top of attempt 2 —
   nets out flat-to-negative vs. baseline despite both sources individually
   being real, human-annotated NER data; not used in the final config.

### Attempt 1 — unlabeled tahrirchi text (abandoned)

Code: [`external_corpora.py`](external_corpora.py).

Fetched one parquet shard each (pinned filenames, via
`huggingface_hub.hf_hub_download`, cached locally) from `tahrirchi/uz-books-v2`
(Cyrillic + Latin fiction/textbooks) and `tahrirchi/uz-crawl` (news + Telegram
posts), split every document into paragraph/sentence-scale chunks matching
`data/train.jsonl`'s length distribution (median 151, p90 1,091 chars — not
whole books or whole articles), and appended a deterministic sample (1,500
chunks each of `books-cyr`, `books-lat`, `crawl-news`, `crawl-telegram`,
seed 42) to train as new records with `entities: []`. `data/train.jsonl`
(13,000) → `train_external.jsonl` (19,000).

**The problem:** these corpora have no NER annotations. Labeling every added
chunk `entities: []` tells the model "zero ORG/NAME/GEO mentions here" — false
for real book and news text, which visibly contains institution names
(`Toshkent kimyo-texnologiya instituti`), countries (`Saudiya Arabistoni`),
people (`Jon Kerri`, `Viktor Hyugo`) — all unlabeled. That's systematic
false-negative label noise, not a verified negative set, and it shows up
directly in the result:

| scope | metric | baseline | +unlabeled corpora | delta |
|---|---|---:|---:|---:|
| micro | precision | 0.7627 | 0.7644 | +0.0017 |
| micro | recall | 0.5744 | 0.5622 | **-0.0122** |
| micro | f1 | 0.6553 | 0.6479 | **-0.0074** |

Recall drops, exactly as the false-negative-noise hypothesis predicts —
precision barely moves, but the model becomes measurably more reluctant to
tag entities after training on thousands of examples where real entities were
marked absent. Net f1 is worse than baseline. Superseded by attempt 2 below;
kept as a documented negative result and as the reason attempt 2 requires
*actual* annotations instead of raw text.

### Attempt 2 — Kaggle courpusNER2015 (recommended)

Code: [`external_ner_datasets.py`](external_ner_datasets.py).

Uses `orvile/named-entity-recognition-for-uzbek-language` on Kaggle
("courpusNER2015", 11,625 BIOES-tagged sentences, PER/ORG/LOC), fetched
anonymously via `kagglehub` (no API key needed — it's a public dataset).
Tokens are rejoined with single spaces (the source has no punctuation tokens)
and BIOES spans decoded into our schema: `PER`→`NAME`, `LOC`→`GEO`,
`ORG`→`ORG`.

```
python -m augmentation.external_ner_datasets \
  --train data/train.jsonl \
  --output augmentation/data/train_external_ner.jsonl \
  --stats augmentation/data/external_ner_stats.json
```

`data/train.jsonl` (13,000) → `train_external_ner.jsonl` (24,625): all 11,625
Kaggle sentences added (428 of them with zero entities — genuine negatives
this time, since the whole corpus was actually annotated, not skipped).

**Other three sources the user pointed at, and why they aren't here (two
evaluated fresh in this step; see attempt 3 for the one that is used, just
not in the recommended config):**

- [`risqaliyevds/uzbek_ner`](https://huggingface.co/datasets/risqaliyevds/uzbek_ner)
  (Hugging Face, 19,609 records): entities are bare mention strings with no
  character offsets, and — the disqualifying finding — 94.3% of records list
  a variant of `O'zbekiston` as a `GPE` entity, while only 21.1% of records
  even contain that string in the text (independently re-verified for this
  step; an earlier pass found `93.8%`/`20.6%` with a slightly stricter
  string match — same conclusion either way). That is templated/hallucinated
  auto-labeling leaking a few-shot example into most outputs, not a reliable
  annotation to train exact-span NER on. Skipped rather than partially
  salvaged.
- [Mendeley 7d59mk8xp5](https://data.mendeley.com/datasets/7d59mk8xp5/1)
  ("Dataset of Uzbek language NER (3000+)", 3,053 sentences, BIOES,
  hand-annotated, CC BY 4.0): well-described and would have been a good
  candidate, but its download bucket
  (`prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com`) returned
  `AccessDenied` to every unauthenticated request tried — direct, with
  `Referer`/`Origin` headers, and via the dataset API's own
  `.../zip?version=1` signed-URL endpoint (which returns a bare URL that
  itself 403s). It appears to require a session Mendeley's site holds
  in-browser, not something scriptable from here — unlike its sibling
  7bxcj57xdz, which the user fetched by hand (see attempt 3). If you can
  download this zip too, hand me the file and `external_ner_datasets.py`
  can add a loader for it directly.

**Data quality found and fixed in the Kaggle corpus itself:**

- Its PER/ORG/LOC tags turned out to cover more ground than our
  proper-noun-only ORG/NAME/GEO definitions (see `LABELING_GUIDE.md`) —
  spot-checking surfaced generic common nouns tagged as entities:
  `tashkilotlarni` ("organizations") and `hokimiyat organlari` ("authority
  bodies") as `ORG`, `talabalar` ("students") and `o'quvchilarni` ("pupils")
  as `PER`, `davlat` ("state") as `LOC`. Every genuine entity in this
  project's own data is a proper noun, capitalized in both scripts, so
  `tokens_to_record` drops any decoded entity whose mention doesn't start
  with an uppercase letter. This is a cheap, precision-favoring heuristic,
  not a semantic fix — it caught 1,725 of 24,453 decoded entities (7.1%) and
  left the surviving mentions visibly clean on inspection (`Iqtisodiyot
  vazirligi`, `Tashqi ishlar vazirligi`, `Namangan`, `Toshkent`, ...). A
  residual risk remains for capitalized-but-still-generic phrases this filter
  can't catch (e.g. a bare `Respublika`, "the Republic"); not pursued further
  given the time budget for this step.
- One recurring source phrase ("Xalq ta'limi vazirligi", Ministry of Public
  Education, appearing in 429/11,625 sentences) had its apostrophe mangled
  into the 3-character mojibake sequence `вЂ™` (a UTF-8 apostrophe
  re-decoded as CP1251 and re-saved) — fixed with a targeted string
  replacement before offsets are computed.

### Training and evaluation

```
python -m baseline.train \
  --train augmentation/data/train_external_ner.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/aug_external_ner

python -m baseline.predict \
  --model-dir artifacts/aug_external_ner/model \
  --input data/dev.jsonl \
  --output artifacts/aug_external_ner/dev_predictions.jsonl

python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/aug_external_ner/dev_predictions.jsonl \
  --output artifacts/aug_external_ner/dev_metrics.json
```

Ran on 24,625 train records (13,000 original + 11,625 Kaggle sentences), 3
epochs, same seed/hyperparameters as `artifacts/baseline`. `best_dev_loss`
improved from 0.1097 (epoch 2 of baseline) to 0.1102 — essentially flat, unlike
step 1's clearer loss improvement. Exact-span dev metrics,
`artifacts/baseline/dev_metrics.json` vs.
`artifacts/aug_external_ner/dev_metrics.json`:

| scope | metric | baseline | +external NER | delta |
|---|---|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6483 | +0.0004 |
| NAME | f1 | 0.6062 | 0.6268 | +0.0206 |
| GEO | f1 | 0.7015 | 0.7153 | +0.0138 |
| micro | precision | 0.7627 | 0.7780 | +0.0154 |
| micro | recall | 0.5744 | 0.5824 | +0.0079 |
| micro | f1 | 0.6553 | 0.6661 | +0.0108 |
| macro | f1 | 0.6518 | 0.6635 | +0.0116 |

Net positive, every label flat-to-better and precision/recall both up, but the
gain is concentrated in NAME (+2.1 f1) and GEO (+1.4 f1) while ORG barely
moves (+0.04 f1). Plausible reading: the Kaggle corpus's PER/LOC entities map
onto NAME/GEO cleanly and add real, varied new surface forms, but its ORG
mentions were the ones most heavily thinned out by the uppercase-only filter
(institutional/authority phrases like `hokimiyat organlari` are exactly the
generic-noun pattern that filter exists to drop), so the ORG training signal
gained the least new material. Smaller than step 1's transliteration gain
(+1.65 micro f1) but a clean win with no observed downside, unlike attempt 1
above.

### Pooled vs. cascaded training

Everything above trains **pooled**: target and external records shuffled
together into one train file, all present in every epoch. The alternative is
**cascaded** (sequential) fine-tuning — first fine-tune on the external
corpus alone, then continue fine-tuning that checkpoint on the target data
alone — which is the standard transfer-learning recipe for exactly this kind
of "different annotator, different conventions" external data: the final
stage's gradient steps are the ones that get the last word, so they pull the
model back toward the target's own conventions instead of averaging them
with the external corpus's for the whole run. Tested here to see whether it
beats the pooled result above.

```
python -m baseline.train \
  --train augmentation/data/train_external_only.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/aug_cascade_stage1

python -m baseline.train \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --model-name artifacts/aug_cascade_stage1/model \
  --output-dir artifacts/aug_cascade_stage2
```

`--model-name` is passed straight to `AutoModelForTokenClassification.from_pretrained`,
which accepts a local checkpoint directory exactly like a hub model id — no
code changes needed for stage 2 to resume from stage 1's weights.
`augmentation/data/train_external_only.jsonl` is the 11,625 Kaggle sentences
from attempt 2 with the 13,000 target records filtered back out (by
`kaggle-uzner2015-` hash prefix), so stage 1 never sees target data. Both
stages use baseline's default 3 epochs / seed 42 / lr 5e-5 — i.e. this is the
"pretrain like a full run, then fine-tune like a full run" version, not a
tuned split; a shorter stage 2 (or lower stage-2 lr) was not tried.

Stage 1 alone, evaluated on dev purely as a sanity check (it never sees
target-style text): micro-F1 0.27 — far below baseline, as expected from a
model that has only ever seen the external corpus's distribution.

Stage 2 (the actual cascaded result) vs. baseline vs. the pooled result from
attempt 2 above:

| scope | metric | baseline | cascaded (stage 1→2) | pooled (attempt 2) |
|---|---|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6433 | 0.6483 |
| NAME | f1 | 0.6062 | 0.6175 | 0.6267 |
| GEO | f1 | 0.7015 | 0.7017 | 0.7151 |
| micro | precision | 0.7627 | 0.7591 | 0.7775 |
| micro | recall | 0.5744 | 0.5791 | 0.5825 |
| micro | f1 | 0.6553 | 0.6570 | 0.6660 |
| macro | f1 | 0.6518 | 0.6542 | 0.6634 |

Cascading barely clears baseline (+0.0017 micro-F1) and is clearly worse
than pooling (-0.0090). `training_summary.json`'s loss curve explains why:
stage 2's `best_dev_loss` (0.1098) and full 3-epoch trajectory are nearly
identical to plain baseline's (0.1097, near-identical shape epoch-by-epoch)
— three full epochs of target-only fine-tuning essentially overwrite
whatever stage 1 learned from the external corpus, i.e. catastrophic
forgetting, the risk this approach trades off against convention-correction
in the first place. **Pooled training (attempt 2) stays the recommended
config for this step.** A shorter stage 2 (e.g. 1 epoch, or a lower learning
rate) would very plausibly retain more of stage 1's signal and might close
the gap to pooled or beat it — not tried here, since it turns this from a
one-shot comparison into a small hyperparameter search.

### Attempt 3 — + Mendeley Gold, combined (tested, not used)

Code: same [`external_ner_datasets.py`](external_ner_datasets.py), via its
`--gold-tsv` option.

The user separately supplied [Mendeley 7bxcj57xdz](https://data.mendeley.com/datasets/7bxcj57xdz/1)
("Dataset of Uzbek language NER (3000+)" gold set, 4,176 sentences, BIO, 8
entity types, CC BY 4.0) as `augmentation/Uzbek_NER_Gold.tsv` — fetched by
hand, since (like its sibling 7d59mk8xp5 above) its own automated download
returns `AccessDenied` to every unauthenticated request. BIO spans are
decoded the same way as Kaggle's (`bio_tokens_to_record`, same
uppercase-mention filter — 612 of 6,103 decoded entities dropped, 10.0%,
noticeably higher than Kaggle's 7.1%) and mapped with the same
`PER`→`NAME`, `LOC`→`GEO`, `ORG`→`ORG`; the other 5 of its 8 tag types
(`MISC`/`TEMPORAL`/`NUMERIC`/`WORK`/`MONEY`) fall outside our schema and are
left untagged.

```
python -m augmentation.external_ner_datasets \
  --train data/train.jsonl \
  --gold-tsv augmentation/Uzbek_NER_Gold.tsv \
  --output augmentation/data/train_external_ner_with_gold.jsonl \
  --stats augmentation/data/external_ner_with_gold_stats.json
```

`data/train.jsonl` (13,000) → `train_external_ner_with_gold.jsonl` (28,801):
attempt 2's 11,625 Kaggle sentences + all 4,176 Mendeley Gold sentences
(1,235 of them with zero entities).

```
python -m baseline.train \
  --train augmentation/data/train_external_ner_with_gold.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/aug_external_ner_with_gold

python -m baseline.predict \
  --model-dir artifacts/aug_external_ner_with_gold/model \
  --input data/dev.jsonl \
  --output artifacts/aug_external_ner_with_gold/dev_predictions.jsonl

python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/aug_external_ner_with_gold/dev_predictions.jsonl \
  --output artifacts/aug_external_ner_with_gold/dev_metrics.json
```

Ran on 28,801 train records, 3 epochs, same seed/hyperparameters as
`artifacts/baseline`. Exact-span dev metrics vs. baseline and vs. attempt 2
(Kaggle only):

| scope | metric | baseline | +Kaggle (attempt 2) | +Kaggle+Gold (attempt 3) | delta vs. baseline |
|---|---|---:|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6483 | 0.6330 | **-0.0148** |
| NAME | f1 | 0.6062 | 0.6268 | 0.6118 | +0.0056 |
| GEO | f1 | 0.7015 | 0.7153 | 0.7078 | +0.0063 |
| micro | precision | 0.7627 | 0.7780 | 0.7826 | +0.0199 |
| micro | recall | 0.5744 | 0.5824 | 0.5626 | **-0.0118** |
| micro | f1 | 0.6553 | 0.6661 | 0.6546 | **-0.0007** |
| macro | f1 | 0.6518 | 0.6635 | 0.6509 | -0.0009 |

Adding Mendeley Gold on top of Kaggle erases attempt 2's entire gain — net
micro f1 is a wash vs. baseline (essentially unchanged) and clearly worse
than attempt 2 alone on every metric except raw precision. The regression is
concentrated in ORG (-1.5 f1 vs. baseline, -1.5 f1 vs. attempt 2). To check
whether this is a combination effect or Mendeley Gold itself, it was also
tried in isolation (baseline + Mendeley Gold only, no Kaggle, 17,176
records, same hyperparameters):

| scope | metric | baseline | +Gold only |
|---|---|---:|---:|
| ORG | f1 | 0.6478 | 0.6351 (**-0.0127**) |
| NAME | f1 | 0.6062 | 0.6215 (+0.0153) |
| GEO | f1 | 0.7015 | 0.7057 (+0.0042) |
| micro | f1 | 0.6553 | 0.6564 (+0.0011) |

Mendeley Gold alone is already roughly neutral (+0.0011 micro f1, well below
attempt 2's +0.0108) and already shows the same ORG regression on its own
(-0.0127, almost identical in size to the -0.0148 seen combined) — so this
is not a Kaggle+Gold interaction, it's a property of the Gold corpus's ORG
annotations specifically. A likely cause: Gold's ORG spans are looser than
this project's convention even after the uppercase filter — e.g. kept
mentions like `1-politexnikumi` or `", LTD kompaniyasining"` (apostrophe/
quote-adjacent tokenization artifacts from the source TSV) that don't match
how `LABELING_GUIDE.md` draws ORG boundaries, teaching the model
inconsistent span edges for exactly the label attempt 2 was already weakest
on.

**Conclusion:** attempt 2 (Kaggle only) is the config used for this step —
`augmentation/data/train_external_ner.jsonl` and `artifacts/aug_external_ner`
reflect it. Attempt 3's combined dataset and model are kept as
`train_external_ner_with_gold.jsonl` / `artifacts/aug_external_ner_with_gold`
for the record, not as the recommended output — a real, documented result
that "more annotated data" isn't automatically better when the second
source's tagging conventions don't match the target schema as closely as
hoped, same spirit as attempt 1's negative result above.

## Step 3 — synthetic augmentation

Code: [`synthetic_augmentation.py`](synthetic_augmentation.py).

### What it does

Adds one noised duplicate of every train record: light character-level
typo noise (keyboard-adjacent substitution, deletion, duplication, adjacent
transposition), each op gated independently per character. Entity spans stay
exactly aligned by construction — the augmented text is built the same way
as in `transliteration.py` (a per-character output "piece" list + prefix-sum
offset array) and the two length-changing ops (delete, duplicate) are
restricted to characters *outside* every gold entity span. Inside a span,
only same-length substitution is allowed, at a lower default rate
(`--entity-prob 0.02` vs. `--free-prob 0.06`), so a span's `start`/`end`
never has to move — only its content is occasionally lightly misspelled,
which is useful on its own (recognizing a typo'd `Тoshkent` as GEO).

```
python -m augmentation.synthetic_augmentation \
  --input data/train.jsonl \
  --output augmentation/data/train_synthetic.jsonl \
  --stats augmentation/data/synthetic_augmentation_stats.json
```

`data/train.jsonl` (13,000) → `augmentation/data/train_synthetic.jsonl`
(26,000: one noised duplicate per record). No transliteration, no external
corpora — only `data/train.jsonl`, so this step's effect is attributable to
the typo noise alone.

**Why not the `sage` package** ([Pomelkin/sage](https://github.com/Pomelkin/sage),
following ai-forever/sage): the operation set implemented here — substitute /
delete / duplicate / transpose, each at a small per-character rate — is the
same rule-based typo-simulation approach SAGE itself uses. SAGE's
higher-fidelity option (`SBSCConfig`) is a statistical model of *real*
spelling-error frequencies, but it's trained on a Russian error corpus with
no Uzbek equivalent available, and it isn't offset-aware for NER spans in the
first place — this step would still need the entity-safe remapping above on
top of it. Given that, the rule-based path is implemented directly rather
than adding the dependency.

### Training and evaluation

```
python -m baseline.train \
  --train augmentation/data/train_synthetic.jsonl \
  --dev data/dev.jsonl \
  --output-dir artifacts/aug_synthetic

python -m baseline.predict \
  --model-dir artifacts/aug_synthetic/model \
  --input data/dev.jsonl \
  --output artifacts/aug_synthetic/dev_predictions.jsonl

python -m evaluation.evaluate_model \
  --gold data/dev.jsonl \
  --predictions artifacts/aug_synthetic/dev_predictions.jsonl \
  --output artifacts/aug_synthetic/dev_metrics.json
```

### Results

Ran on 26,000 train records (13,000 original + 13,000 noised duplicates), 3
epochs, same seed/hyperparameters as `artifacts/baseline`. `best_dev_loss`
was actually reached at epoch 1 (0.1176) and got worse every epoch after
(0.1242, 0.1426) — the noised duplicates make the model converge to a worse
generalizing point faster rather than helping it generalize better. Exact-span
dev metrics, `artifacts/baseline/dev_metrics.json` vs.
`artifacts/aug_synthetic/dev_metrics.json`:

| scope | metric | baseline | +synthetic | delta |
|---|---|---:|---:|---:|
| ORG | f1 | 0.6478 | 0.6187 | **-0.0292** |
| NAME | f1 | 0.6062 | 0.6094 | +0.0031 |
| GEO | f1 | 0.7015 | 0.6992 | -0.0023 |
| micro | precision | 0.7627 | 0.7519 | -0.0108 |
| micro | recall | 0.5744 | 0.5640 | -0.0104 |
| micro | f1 | 0.6553 | 0.6445 | **-0.0108** |
| macro | f1 | 0.6518 | 0.6424 | -0.0095 |

Net negative, driven mostly by ORG (-2.9 f1 points), with GEO roughly flat and
NAME marginally better. A plausible reading: ORG mentions in this dataset are
disproportionately multi-word official names (`Tashqi ishlar vazirligi`-style
strings), so they have the most surface area for typo noise to land on and
the least redundancy to survive it, while short GEO/NAME tokens are more
robust to a stray character. Doubling the effective train set with a noisier
version of the same 13,000 sentences (rather than genuinely new text, as in
step 1's script-flipped duplicates) may also just be adding more overfitting
pressure than regularization at these default noise rates. Lowering
`--free-prob`/`--entity-prob` or dropping `--copies-per-record` would be the
next things to try, not attempted here to keep this step's result to the
configuration described above.
