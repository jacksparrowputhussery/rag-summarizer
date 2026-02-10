# evaluation/metrics.py

def recall_at_k(docs, evidence, k):
    if not evidence:
        return 1.0
    hits = 0
    for ev in evidence:
        for d in docs[:k]:
            if ev.lower() in d.content.lower():
                hits += 1
                break
    return hits / len(evidence)


def mrr(docs, evidence):
    for i, d in enumerate(docs, 1):
        for ev in evidence:
            if ev.lower() in d.content.lower():
                return 1.0 / i
    return 0.0
