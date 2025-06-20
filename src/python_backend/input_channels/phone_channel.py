# phone_channel.py - Telefon/Voice Input Channel
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .base_input_channel import InputChannel
from .input_message import InputMessage

logger = logging.getLogger(__name__)

class PhoneChannel(InputChannel):
    """Telefon/Voice Input Channel"""
    
    def __init__(self, voice_config: Dict[str, str] = None):
        super().__init__("Phone")
        self.voice_config = voice_config or {}
    
    async def initialize(self) -> bool:
        """Initialisiert Voice Recognition"""
        logger.info("Initialisiere Phone/Voice Channel...")
        # Hier würde Speech-to-Text initialisiert
        self.active = True
        logger.info("Phone Channel bereit (Simulation)")
        return True
    
    async def process_input(self, raw_input: Dict[str, Any]) -> Optional[InputMessage]:
        """Verarbeitet Sprachnachricht"""
        try:
            # Simuliere Speech-to-Text
            transcript = raw_input.get('transcript', '')
            confidence = raw_input.get('confidence', 0.0)
            phone_number = raw_input.get('phone_number', 'unknown')
            duration = raw_input.get('duration_seconds', 0)
            timestamp = datetime.fromisoformat(raw_input.get('timestamp', datetime.now().isoformat()))
            
            # Bereinige Transkript
            processed_text = self._clean_transcript(transcript)
            
            # Erstelle Metadata
            metadata = {
                'phone_number': phone_number,
                'transcript_confidence': confidence,
                'call_duration': duration,
                'language': raw_input.get('language', 'de'),
                'background_noise': raw_input.get('background_noise', False)
            }
            
            # Sprachanrufe haben oft höhere Priorität
            priority = self.extract_priority_indicators(processed_text)
            if duration < 30:  # Kurze Anrufe sind oft dringend
                priority = max(priority, 1)
            
            self.message_count += 1
            
            return InputMessage(
                channel=self.name,
                raw_content=transcript,
                processed_content=processed_text,
                metadata=metadata,
                timestamp=timestamp,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Voice Verarbeitung fehlgeschlagen: {e}")
            return None
    
    def _clean_transcript(self, transcript: str) -> str:
        """Bereinigt Speech-to-Text Transkript"""
        # Entferne Füllwörter
        filler_words = ['ähm', 'äh', 'hmm', 'also', 'ja also', 'sozusagen']
        
        cleaned = transcript.lower()
        for filler in filler_words:
            cleaned = cleaned.replace(f' {filler} ', ' ')
        
        # Korrigiere häufige STT-Fehler
        corrections = {
            'aufzug': 'aufzug',
            'auf zug': 'aufzug',
            'fahr kabine': 'fahrkabine',
            'not ruf': 'notruf'
        }
        
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        
        return cleaned.strip()