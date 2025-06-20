# sms_channel.py - SMS Input Channel
import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime
from .base_input_channel import InputChannel
from .input_message import InputMessage

logger = logging.getLogger(__name__)

class SMSChannel(InputChannel):
    """SMS Input Channel"""
    
    def __init__(self, api_config: Dict[str, str] = None):
        super().__init__("SMS")
        self.api_config = api_config or {}
    
    async def initialize(self) -> bool:
        """Initialisiert SMS Gateway"""
        logger.info("Initialisiere SMS Channel...")
        # Hier würde echte SMS-API initialisiert
        self.active = True
        logger.info("SMS Channel bereit (Simulation)")
        return True
    
    async def process_input(self, raw_input: Dict[str, Any]) -> Optional[InputMessage]:
        """Verarbeitet SMS-Nachricht"""
        try:
            # Extrahiere SMS-Daten
            phone_number = raw_input.get('from', 'unknown')
            text = raw_input.get('text', '')
            timestamp = datetime.fromisoformat(raw_input.get('timestamp', datetime.now().isoformat()))
            
            # Bereinige und normalisiere Text
            processed_text = self._normalize_sms_text(text)
            
            # Erstelle Metadata
            metadata = {
                'phone_number': phone_number,
                'original_length': len(text),
                'carrier': raw_input.get('carrier', 'unknown')
            }
            
            self.message_count += 1
            
            return InputMessage(
                channel=self.name,
                raw_content=text,
                processed_content=processed_text,
                metadata=metadata,
                timestamp=timestamp,
                priority=self.extract_priority_indicators(processed_text)
            )
            
        except Exception as e:
            logger.error(f"SMS Verarbeitung fehlgeschlagen: {e}")
            return None
    
    def _normalize_sms_text(self, text: str) -> str:
        """Normalisiert SMS-Text"""
        # Entferne SMS-Kürzel
        replacements = {
            'lg': 'liebe grüße',
            'mfg': 'mit freundlichen grüßen',
            'asap': 'so schnell wie möglich',
            'btw': 'übrigens'
        }
        
        normalized = text.lower()
        for abbr, full in replacements.items():
            normalized = normalized.replace(abbr, full)
        
        # Entferne mehrfache Leerzeichen
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized