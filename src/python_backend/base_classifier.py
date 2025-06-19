# base_classifier.py - Basis Klassifikator Interface
from abc import ABC, abstractmethod
from typing import Dict, List
from event import Event, ClassificationResult
import time

class BaseClassifier(ABC):
    """Abstrakte Basis für alle Klassifikatoren"""
    
    def __init__(self, name: str):
        self.name = name
        self.total_classifications = 0
        self.total_time = 0.0
    
    @abstractmethod
    async def classify(self, event: Event) -> ClassificationResult:
        """Klassifiziert ein Event - muss implementiert werden"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialisiert den Klassifikator - muss implementiert werden"""
        pass
    
    def get_stats(self) -> Dict[str, float]:
        """Gibt Statistiken zurück"""
        avg_time = self.total_time / max(1, self.total_classifications)
        return {
            'total_classifications': self.total_classifications,
            'total_time': self.total_time,
            'average_time': avg_time
        }
    
    async def classify_with_timing(self, event: Event) -> ClassificationResult:
        """Wrapper mit Timing"""
        start_time = time.time()
        result = await self.classify(event)
        processing_time = time.time() - start_time
        
        self.total_classifications += 1
        self.total_time += processing_time
        
        # Update processing time in result
        result.processing_time = processing_time
        result.classifier_name = self.name
        
        return result