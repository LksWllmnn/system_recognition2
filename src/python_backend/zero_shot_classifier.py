# zero_shot_classifier.py - Zero-Shot Klassifikator über HuggingFace
import logging
from base_classifier import BaseClassifier
from event import Event, ClassificationResult

from transformers import pipeline

logger = logging.getLogger(__name__)

class ZeroShotClassifier(BaseClassifier):
    def __init__(self, candidate_labels=None, model_name="facebook/bart-large-mnli"):
        super().__init__("ZeroShot")
        self.model_name = model_name
        self.labels = candidate_labels or ["seil", "fahrkabine", "aufzugsgetriebe"]
        self.classifier = None

    async def initialize(self) -> bool:
        try:
            logger.info(f"Lade Zero-Shot-Modell: {self.model_name}")
            self.classifier = pipeline("zero-shot-classification", model=self.model_name)
            logger.info("✅ Zero-Shot Klassifikator bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        if not self.classifier:
            return ClassificationResult(event, categories={'unknown': 1.0}, confidence=0.0, processing_time=0.0, classifier_name=self.name)

        result = self.classifier(event.message, self.labels)
        categories = dict(zip(result['labels'], result['scores']))
        best = result['labels'][0]

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=categories[best],
            processing_time=0.0,
            classifier_name=self.name
        )
