# event.py - Event Datenstrukturen
from dataclasses import dataclass, asdict
from typing import Dict, Any
from datetime import datetime

@dataclass
class Event:
    """Basis Event-Datenstruktur"""
    timestamp: datetime
    message: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Event zu Dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Erstellt Event aus Dictionary"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ClassificationResult:
    """Klassifikationsergebnis"""
    event: Event
    categories: Dict[str, float]
    confidence: float
    processing_time: float
    classifier_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            'event': self.event.to_dict(),
            'categories': self.categories,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'classifier_name': self.classifier_name
        }