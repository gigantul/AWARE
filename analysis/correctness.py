# analysis/correctness.py

from sentence_transformers import CrossEncoder
from typing import Dict

class CrossEncoderSimilarity:
    def __init__(self, model_id: str = "cross-encoder/stsb-distilroberta-base", device: str = "cuda"):
        self.model = CrossEncoder(model_id, device=device)

    def score(self, source: str, prediction: str) -> float:
        return self.model.predict([[source, prediction]])[0]

    def score_many(self, sources: list[str], predictions: list[str]) -> list[float]:
        return self.model.predict([[src, pred] for src, pred in zip(sources, predictions)]).tolist()


cross_encoder = CrossEncoderSimilarity()

def evaluate_response(sample: Dict, output: Dict) -> float:
    """
    Evaluate model-generated answer against reference answers using semantic similarity.

    Args:
        sample: contains 'answer' and 'additional_answers'
        output: contains 'generated_text'

    Returns:
        float: max similarity score
    """
    prediction = output["generated_text"].strip().rstrip(".")
    answers = sample.get("answer", [])
    if isinstance(answers, str):
        answers = [answers]
    if sample.get("additional_answers"):
        answers += sample["additional_answers"]
    answers = [a.strip().rstrip(".") for a in answers if isinstance(a, str) and a.strip()]

    if not prediction or not answers:
        return 0.0

    scores = cross_encoder.score_many(sources=answers, predictions=[prediction] * len(answers))
    return max(scores) if scores else 0.0
