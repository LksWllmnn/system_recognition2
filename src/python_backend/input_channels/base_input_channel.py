# base_input_channel.py - Abstrakte Basis für Input-Kanäle
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from .input_message import InputMessage

logger = logging.getLogger(__name__)

class InputChannel(ABC):
    """Abstrakte Basis für Input-Kanäle"""
    
    def __init__(self, name: str):
        self.name = name
        self.active = False
        self.message_count = 0
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialisiert den Kanal"""
        pass
    
    @abstractmethod
    async def process_input(self, raw_input: Any) -> Optional[InputMessage]:
        """Verarbeitet Roh-Input zu vereinheitlichter Nachricht"""
        pass
    
    def extract_priority_indicators(self, text: str) -> int:
        """Extrahiert Priorität aus Text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['notfall', 'emergency', 'sofort', 'kritisch']):
            return 2
        elif any(word in text_lower for word in ['wichtig', 'dringend', 'schnell']):
            return 1
        return 0