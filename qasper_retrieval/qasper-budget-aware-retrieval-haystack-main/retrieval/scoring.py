# retrieval/scoring.py

from haystack import Document


def normalize(vals):
    if not vals:
        return vals
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def position_prior(p: float) -> float:
    d = min(p, 1 - p)
    return 1.0 - 0.4 * (2 * d) ** 2


def score_documents(candidates, signal_weights, section_weights):
    sem = normalize([c["semantic"] for c in candidates])
    lex = normalize([c["lexical"] for c in candidates])

    scored = []

    max_struct = max(section_weights.values())

    for i, c in enumerate(candidates):
        doc = c["doc"]
        section = doc.meta.get("section_type", "body")
        struct = section_weights.get(section, 1.0) / max_struct
        pos = position_prior(doc.meta.get("position", 0.5))

        score = (
            signal_weights["semantic"] * sem[i]
            + signal_weights["lexical"] * lex[i]
            + signal_weights["structural"] * struct
            + signal_weights["position"] * pos
        )

        scored.append(
            Document(
                content=doc.content,
                meta=dict(doc.meta),
                score=score,
                id=doc.id,
            )
        )

    return sorted(scored, key=lambda d: d.score, reverse=True)
