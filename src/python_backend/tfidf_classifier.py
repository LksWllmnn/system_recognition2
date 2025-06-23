# tfidf_classifier.py - Klassischer ML-Klassifikator basierend auf TF-IDF
import logging
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from base_classifier import BaseClassifier
from event import Event, ClassificationResult

logger = logging.getLogger(__name__)

class TfidfMLClassifier(BaseClassifier):
    def __init__(self, training_data: Dict[str, str]):
        super().__init__("TfidfML")
        self.pipeline = None
        self.training_data = training_data  # Dict[text] = category

    async def initialize(self) -> bool:
        logger.info("Initialisiere TF-IDF Klassifikator...")
        try:
            texts = list(self.training_data.keys())
            labels = list(self.training_data.values())

            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                ('clf', LogisticRegression(max_iter=200))
            ])
            self.pipeline.fit(texts, labels)
            logger.info("✅ TF-IDF Klassifikator bereit")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren: {e}")
            return False

    async def classify(self, event: Event) -> ClassificationResult:
        if not self.pipeline:
            return ClassificationResult(event, categories={'unknown': 1.0}, confidence=0.0, processing_time=0.0, classifier_name=self.name)

        pred = self.pipeline.predict_proba([event.message])[0]
        labels = self.pipeline.classes_

        categories = dict(zip(labels, pred))
        best = max(categories, key=categories.get)

        return ClassificationResult(
            event=event,
            categories=categories,
            confidence=categories[best],
            processing_time=0.0,
            classifier_name=self.name
        )
