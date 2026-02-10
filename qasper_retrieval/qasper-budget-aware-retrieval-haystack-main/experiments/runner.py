# experiments/runner.py

from evaluation.evaluator import QASPEREvaluator


class ExperimentRunner:
    def __init__(self, questions):
        self.questions = questions
        self.evaluator = QASPEREvaluator()

    def run(self, retriever):
        return self.evaluator.evaluate(retriever, self.questions)
