"""Strict-span micro-F1 -- та же метрика, что у организаторов.

Сущность считается верной, только если совпали ВСЕ ТРИ: start, end, label.
Никакого partial credit. Плюс разрезы по классу и по графике документа.
"""
from collections import Counter, defaultdict
from .boundary_kit.audit import script_of


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(gold_recs, pred_by_id, by_script=True):
    """gold_recs: [{'id','text','entities':[{start,end,type}]}]
    pred_by_id: {id: [Span|dict]}  -> метрики + список ошибок для разбора."""
    TP = FP = FN = 0
    per_cls = defaultdict(lambda: [0, 0, 0])
    per_scr = defaultdict(lambda: [0, 0, 0])
    errors = {"boundary_only": [], "type_only": [], "spurious": [], "missed": []}

    for r in gold_recs:
        g = {(e["start"], e["end"], e["type"]) for e in r["entities"]}
        praw = pred_by_id.get(r["id"], [])
        p = {(x.start, x.end, x.type) if hasattr(x, "start")
             else (x["start"], x["end"], x["type"]) for x in praw}
        scr = script_of(r["text"]) if by_script else "all"

        for t in g & p:
            TP += 1; per_cls[t[2]][0] += 1; per_scr[scr][0] += 1
        for t in p - g:
            FP += 1; per_cls[t[2]][1] += 1; per_scr[scr][1] += 1
            # классификация ошибки: границы vs тип vs выдумка
            same_bounds = [x for x in g if x[0] == t[0] and x[1] == t[1]]
            overlap = [x for x in g if not (t[1] <= x[0] or t[0] >= x[1])]
            if same_bounds:
                errors["type_only"].append((r["id"], t, same_bounds[0], r["text"][t[0]:t[1]]))
            elif overlap:
                errors["boundary_only"].append((r["id"], t, overlap[0],
                                                r["text"][t[0]:t[1]],
                                                r["text"][overlap[0][0]:overlap[0][1]]))
            else:
                errors["spurious"].append((r["id"], t, r["text"][t[0]:t[1]]))
        for t in g - p:
            FN += 1; per_cls[t[2]][2] += 1; per_scr[scr][2] += 1
            if not any(not (t[1] <= x[0] or t[0] >= x[1]) for x in p):
                errors["missed"].append((r["id"], t, r["text"][t[0]:t[1]]))

    out = {"micro": dict(zip("PRF", prf(TP, FP, FN))), "TP": TP, "FP": FP, "FN": FN,
           "by_class": {k: dict(zip("PRF", prf(*v))) for k, v in per_cls.items()},
           "by_script": {k: dict(zip("PRF", prf(*v))) for k, v in per_scr.items()},
           "error_counts": {k: len(v) for k, v in errors.items()},
           "errors": errors}
    return out
