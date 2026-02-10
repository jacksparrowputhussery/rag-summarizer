# evaluation/evaluator.py

from evaluation.metrics import recall_at_k, mrr
from tqdm import tqdm
import numpy as np


class QASPEREvaluator:
    def evaluate(self, retriever, questions, k_values=[5, 10]):
        results = {f"recall@{k}": [] for k in k_values}
        results["mrr"] = []

        for q in tqdm(questions, desc="Evaluating"):
            docs = retriever.retrieve(
                q["question"],
                filters={"paper_id": [q["paper_id"]]},
                top_k=max(k_values),
            )

            for k in k_values:
                results[f"recall@{k}"].append(
                    recall_at_k(docs, q["evidence"], k)
                )

            results["mrr"].append(mrr(docs, q["evidence"]))

        return {
            k: (float(np.mean(v)), float(np.std(v)))
            for k, v in results.items()
        }
