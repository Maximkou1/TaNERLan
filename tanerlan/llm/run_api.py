"""Run an OpenAI-compatible (vLLM) chat endpoint over dev.jsonl via the llm_ner harness."""
from __future__ import annotations
import argparse, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm_ner import build_system_prompt, build_user_prompt, align_surfaces, parse_json_lenient, pick_shots

# Majority boundary conventions from tanerlan/llm/boundary_audit_train.md.
CONVENTION = {
    "suffix_included": True,
    "quote_included": False,
    "geotail_included": True,
    "title_included": False,
}

# Additive rule blocks for prompt-tuning experiments (see tanerlan/llm/PROMPT_TUNING.md).
# Selected via --rules a,b,c and appended to build_system_prompt()'s output.
EXTRA_RULES = {
    "lowercase": (
        "- Do NOT rely on capitalization as a signal. In this casual "
        "social-media text, real names/places/brands are often written "
        "fully lowercase: 'oltiariq' (a place), 'muzaffar' (a person), "
        "'pepsi' (a brand), 'uzbekistan' (a country) are all valid "
        "entities despite the lowercase spelling. Judge by meaning, not "
        "by capital letters."
    ),
    "everyoccurrence": (
        "- If an entity string appears N times in the text, you MUST "
        "output it N separate times (once per occurrence), even though "
        "the text and label are identical each time -- do not deduplicate."
    ),
    "handles": (
        "- Social-media handle/username-shaped tokens are NOT entities, "
        "even when a real name/brand/place is embedded inside them. "
        "Recognize the shape: a single token (no internal space) that is "
        "either ALL CAPS with a trailing number ('MIAMIOPEN', 'UFC327' as "
        "one glued word), snake_case or ends in a common suffix like "
        "'_uz'/'_design' ('BaxtliOilam_uz', 'Madinaxon_design', "
        "'S_U_Network'), glues 2+ capitalized words with no separator "
        "('USUBasketball', 'VinnysCorner1', 'AmirxonValijon'), or is an "
        "all-lowercase run-on word that is not a real dictionary word "
        "('aaaronsmithhh', 'samandarbro23'). Skip these entirely -- they "
        "are account names, not the entities they might be about. A "
        "normally-spelled name/brand with internal spaces or normal "
        "capitalization ('Alisher Navoiy', 'Real Madrid') is unaffected."
    ),
    "notagsalad": (
        "- Do NOT extract literal #hashtags or @handles as entities, and "
        "do not extract made-up portmanteau/compound tag-words formed by "
        "gluing two brand/place words together with no space (e.g. "
        "'jeepbrasil', 'fordjeep' in '#jeep willys jeepbrasil #4x4 "
        "fordjeep' -> only 'jeep' is a real brand mention, the glued "
        "portmanteaus are decorative tags, not entities). A real standalone "
        "word that names a brand/place/person (even in a short caption or "
        "list, e.g. 'instagram', 'SPURS', 'ABC') IS still a valid entity."
    ),
    "tailvocab": (
        "- ALWAYS extend the span to swallow a directly-following "
        "institutional/administrative tail: shahri/shahrida/shahridagi, "
        "tumani/tumanida, viloyati/viloyatida, respublikasi, mahallasi, "
        "koʻchasi, maktabi, universiteti, kasalxonasi, bank/banki, "
        "klub/klubi, jamoasi, terma jamoasi, vazirligi, agentligi, "
        "qoʻmitasi, boshqarmasi, hokimligi, kengashi, markazi, "
        "departamenti, prokuraturasi, xizmati, chempionati, kubogi/kubogida "
        "and Cyrillic equivalents (шаҳри, вилояти, бошқармаси, вазирлиги, "
        "терма жамоаси, чемпионати, кубоги, департаменти, хизмати, ...). "
        "This is a strict rule, not a suggestion: never emit the bare name "
        "alone when one of these words immediately follows it. "
        "'Тошкент шаҳри' -> GEO (whole phrase). "
        "'Жиззах вилоят Соғлиқни сақлаш бошқармаси' -> ORG (whole phrase, "
        "not just 'Жиззах вилоят'). "
        "'Oʻzbekiston milliy terma jamoasida' -> ORG (whole phrase, not "
        "just 'Oʻzbekiston')."
    ),
    "geoorg": (
        "- A place/country name that is followed by, or clearly refers to "
        "in context, a team/club/championship/institution is ORG, not GEO: "
        "'Angliya chempionati' -> ORG; a national team named after its "
        "country ('Oʻzbekiston terma jamoasi', or a football club named "
        "after its town like 'Southport', 'Yeovil' used as a team) -> ORG. "
        "A bare place/country name used only to say where something is or "
        "happened ('Toshkentda', 'Angliyada joylashgan') stays GEO."
    ),
    "nickname": (
        "- Social-media handles, stan nicknames and short lowercase "
        "aliases count as NAME even if they look like ordinary words or "
        "are 2-4 characters (e.g. 'jk', 'jkga', '@handle', 'Tae'). Do not "
        "skip them for looking informal."
    ),
    "recall": (
        "- Prioritize recall: if a substring plausibly matches GEO, NAME "
        "or ORG under these rules, include it. It is better to include a "
        "borderline candidate than to silently drop it."
    ),
    "complete": (
        "- List EVERY entity mention in the text, including ones in "
        "Cyrillic script and ones near the end of a long text. Do not stop "
        "early or summarize -- go through the whole text."
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://151.237.25.16:23343")
    p.add_argument("--model", default="cyankiwi/Qwen3.8-27B-AWQ-INT4")
    p.add_argument("--api-key", default="sk-local")
    p.add_argument("--input", type=Path, default=Path("data/dev.jsonl"))
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"), help="source for few-shot examples")
    p.add_argument("--output", type=Path, default=Path("artifacts/api/dev_predictions.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--thinking", action="store_true", help="leave the model's reasoning mode on")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--rules", default="tailvocab,geoorg",
                    help=f"comma-separated extra rule blocks: {','.join(EXTRA_RULES)} "
                         "(default = best combo found in prompt tuning, see tanerlan/llm/PROMPT_TUNING.md)")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_call_fn(url: str, model: str, api_key: str, max_tokens: int, thinking: bool, timeout: float):
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"

    def call_fn(system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise last_err

    return call_fn


def main() -> None:
    args = parse_args()
    system = build_system_prompt(CONVENTION)
    if args.rules:
        keys = [k.strip() for k in args.rules.split(",") if k.strip()]
        system += "\n" + "\n".join(EXTRA_RULES[k] for k in keys)
    call_fn = make_call_fn(args.url, args.model, args.api_key, args.max_tokens, args.thinking, args.request_timeout)

    shots = None
    if args.shots:
        train_records = load_jsonl(args.train)
        shots = pick_shots(train_records, k=args.shots)

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]

    def predict_one(rec: dict) -> dict:
        text = rec["text"]
        try:
            raw = call_fn(system=system, user=build_user_prompt(text, shots))
            entities = align_surfaces(text, parse_json_lenient(raw))
        except (urllib.error.URLError, OSError, TimeoutError, KeyError, IndexError) as e:
            print(f"  [{rec['hash']}] request failed: {e}")
            entities = []
        entities = [{"label": e["type"], "start": e["start"], "end": e["end"]} for e in entities]
        return {"hash": rec["hash"], "entities": entities}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with args.output.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for i, result in enumerate(pool.map(predict_one, records)):
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"{i + 1}/{len(records)}  ({elapsed / (i + 1):.2f}s/doc avg)")
    print(f"wrote {len(records)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
