# email_channel.py - Email Input Channel
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from .base_input_channel import InputChannel
from .input_message import InputMessage

logger = logging.getLogger(__name__)

class EmailChannel(InputChannel):
    """Email Input Channel"""
    
    def __init__(self, email_config: Dict[str, str] = None):
        super().__init__("Email")
        self.email_config = email_config or {}
    
    async def initialize(self) -> bool:
        """Initialisiert Email Client"""
        logger.info("Initialisiere Email Channel...")
        # Hier würde echter Email-Client initialisiert
        self.active = True
        logger.info("Email Channel bereit (Simulation)")
        return True
    
    async def process_input(self, raw_input: Dict[str, Any]) -> Optional[InputMessage]:
        """Verarbeitet Email"""
        try:
            # Extrahiere Email-Daten
            sender = raw_input.get('from', 'unknown@example.com')
            subject = raw_input.get('subject', '')
            body = raw_input.get('body', '')
            timestamp = datetime.fromisoformat(raw_input.get('timestamp', datetime.now().isoformat()))
            
            # Kombiniere Subject und Body
            full_text = f"{subject}\n{body}"
            
            # Extrahiere relevanten Text
            processed_text = self._extract_email_content(full_text)
            
            # Erstelle Metadata
            metadata = {
                'sender': sender,
                'subject': subject,
                'has_attachments': raw_input.get('attachments', []) != [],
                'thread_id': raw_input.get('thread_id'),
                'is_reply': 'Re:' in subject or 'AW:' in subject
            }
            
            # Erhöhe Priorität für bestimmte Betreffzeilen
            priority = self.extract_priority_indicators(full_text)
            if any(word in subject.lower() for word in ['urgent', 'dringend', 'wichtig']):
                priority = max(priority, 1)
            
            self.message_count += 1
            
            return InputMessage(
                channel=self.name,
                raw_content=full_text,
                processed_content=processed_text,
                metadata=metadata,
                timestamp=timestamp,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Email Verarbeitung fehlgeschlagen: {e}")
            return None
    
    def _extract_email_content(self, text: str) -> str:
        """Extrahiert relevanten Content aus Email"""
        # Entferne Email-Signaturen
        lines = text.split('\n')
        content_lines = []
        
        for line in lines:
            # Stoppe bei typischen Signatur-Markern
            if line.strip() in ['--', '___', '---'] or line.startswith('Von:') or line.startswith('From:'):
                break
            content_lines.append(line)
        
        content = '\n'.join(content_lines)
        
        # Entferne Zitate (lines starting with >)
        content = '\n'.join(line for line in content.split('\n') if not line.strip().startswith('>'))
        
        return content.strip()