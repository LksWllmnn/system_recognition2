
import logging
from typing import Any, Optional
from datetime import datetime
from .base_input_channel import InputChannel
from .input_message import InputMessage

logger = logging.getLogger(__name__)

class DirectInputChannel(InputChannel):
    """Direkte Texteingabe Channel"""
    
    def __init__(self):
        super().__init__("DirectInput")
    
    async def initialize(self) -> bool:
        """Initialisiert Direct Input"""
        logger.info("Initialisiere Direct Input Channel...")
        self.active = True
        return True
    
    async def process_input(self, raw_input: Any) -> Optional[InputMessage]:
        """Verarbeitet direkte Eingabe"""
        try:
            if isinstance(raw_input, str):
                text = raw_input
                metadata = {}
            else:
                text = raw_input.get('text', '')
                metadata = raw_input.get('metadata', {})
            
            timestamp = datetime.now()
            
            self.message_count += 1
            
            return InputMessage(
                channel=self.name,
                raw_content=text,
                processed_content=text.strip(),
                metadata=metadata,
                timestamp=timestamp,
                priority=self.extract_priority_indicators(text)
            )
            
        except Exception as e:
            logger.error(f"Direct Input Verarbeitung fehlgeschlagen: {e}")
            return None# direct_channel.py - Direkte Texteingabe Channel