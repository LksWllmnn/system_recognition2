# emergency_channel.py - Notfallknopf Channel
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .base_input_channel import InputChannel
from .input_message import InputMessage

logger = logging.getLogger(__name__)

class EmergencyButtonChannel(InputChannel):
    """Notfallknopf Channel"""
    
    def __init__(self):
        super().__init__("EmergencyButton")
    
    async def initialize(self) -> bool:
        """Initialisiert Notfallknopf-Interface"""
        logger.info("Initialisiere Emergency Button Channel...")
        self.active = True
        logger.info("Emergency Button Channel bereit")
        return True
    
    async def process_input(self, raw_input: Dict[str, Any]) -> Optional[InputMessage]:
        """Verarbeitet Notfallknopf-Aktivierung"""
        try:
            location = raw_input.get('location', 'unknown')
            button_id = raw_input.get('button_id', 'unknown')
            timestamp = datetime.fromisoformat(raw_input.get('timestamp', datetime.now().isoformat()))
            
            # Generiere Standard-Notfallnachricht
            processed_text = f"NOTFALL: Notfallknopf {button_id} in {location} wurde aktiviert"
            
            # Metadata
            metadata = {
                'button_id': button_id,
                'location': location,
                'floor': raw_input.get('floor'),
                'building': raw_input.get('building'),
                'auto_generated': True
            }
            
            self.message_count += 1
            
            return InputMessage(
                channel=self.name,
                raw_content=f"Emergency button {button_id} pressed",
                processed_content=processed_text,
                metadata=metadata,
                timestamp=timestamp,
                priority=2  # Immer höchste Priorität
            )
            
        except Exception as e:
            logger.error(f"Emergency Button Verarbeitung fehlgeschlagen: {e}")
            return None