# simple_classifier.py - Einfacher Aufzugs-Klassifikator
import re
import logging
from typing import Dict
from base_classifier import BaseClassifier
from event import Event, ClassificationResult

logger = logging.getLogger(__name__)

class SimpleEmbeddingClassifier(BaseClassifier):
    """Einfacher Keyword-basierter Aufzugs-Klassifikator"""
    
    def __init__(self):
        super().__init__("SimpleEmbedding")
        self.categories = {
            'fahrkabine': [
                'tür', 'türe', 'türöffnung', 'türschließung', 'kabine', 'fahrkabine',
                'beleuchtung', 'display', 'knopf', 'taste', 'bedienfeld', 'panel',
                'lüftung', 'ventilation', 'innenraum', 'boden', 'wand', 'decke',
                'türsensor', 'lichtschranke', 'notruf', 'notsprechanlage'
            ],
            'seil': [
                'seil', 'seile', 'tragseil', 'führungsseil', 'hubseil', 'kabel',
                'draht', 'spannung', 'seilführung', 'seilrolle', 'umlenkrolle',
                'seilscheibe', 'aufhängung', 'befestigung', 'seilüberwachung',
                'bruch', 'riss', 'verschleiß', 'dehnung'
            ],
            'aufzugsgetriebe': [
                'getriebe', 'motor', 'antrieb', 'antriebsmotor', 'getriebemotor',
                'öl', 'schmierung', 'schmierstoff', 'schmierölstand', 'getriebeöl',
                'vibration', 'lager', 'welle', 'zahnrad', 'kupplung', 
                'bremse', 'bremsung', 'drehzahl', 'geschwindigkeit', 'drehmoment',
                'temperatur', 'überhitzung', 'kühlung', 'steuerung', 'steuerungseinheit'
            ]
        }
    
    async def initialize(self) -> bool:
        """Initialisiert den Klassifikator"""
        logger.info("Initialisiere Simple Aufzugs-Klassifikator...")
        logger.info("Simple Aufzugs-Klassifikator bereit")
        return True
    
    async def classify(self, event: Event) -> ClassificationResult:
        """Klassifiziert Event nach Aufzugsteilen"""
        message = event.message.lower()
        scores = {}
        
        for category, keywords in self.categories.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword in message:
                    # Gewichtung: Längere Keywords = höhere Relevanz
                    weight = len(keyword) / 10.0
                    score += weight
                    matches += 1
            
            # Bonus für mehrere Treffer
            if matches > 1:
                score *= (1 + matches * 0.1)
            
            scores[category] = min(score, 1.0)  # Maximal 1.0
        
        # Finde beste Kategorie
        if scores:
            best_category = max(scores, key=scores.get)
            confidence = scores[best_category]
        else:
            best_category = 'unknown'
            confidence = 0.0
            scores['unknown'] = 1.0
        
        return ClassificationResult(
            event=event,
            categories=scores,
            confidence=confidence,
            processing_time=0.0,  # Wird später gesetzt
            classifier_name=self.name
        )