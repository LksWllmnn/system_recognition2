# multi_channel_handler.py - Multi-Channel Input Handler
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from collections import defaultdict

from .base_input_channel import InputChannel
from .input_message import InputMessage
from .sms_channel import SMSChannel
from .email_channel import EmailChannel
from .phone_channel import PhoneChannel
from .emergency_channel import EmergencyButtonChannel
from .direct_channel import DirectInputChannel

logger = logging.getLogger(__name__)

class MultiChannelInputHandler:
    """Verwaltet alle Input-Kanäle"""
    
    def __init__(self):
        self.channels: Dict[str, InputChannel] = {}
        self.message_queue = asyncio.Queue()
        self.message_callback: Optional[Callable] = None
        self.total_messages = 0
        self.messages_by_channel = defaultdict(int)
        self.messages_by_priority = defaultdict(int)
    
    async def initialize_all_channels(self):
        """Initialisiert alle verfügbaren Kanäle"""
        logger.info("Initialisiere Multi-Channel Input System...")
        
        # Erstelle alle Kanäle
        channels_to_init = [
            SMSChannel(),
            EmailChannel(),
            PhoneChannel(),
            EmergencyButtonChannel(),
            DirectInputChannel()
        ]
        
        # Initialisiere parallel
        init_tasks = []
        for channel in channels_to_init:
            init_tasks.append(self._init_channel(channel))
        
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Registriere erfolgreiche Kanäle
        for channel, result in zip(channels_to_init, results):
            if isinstance(result, Exception):
                logger.error(f"Kanal {channel.name} Initialisierung fehlgeschlagen: {result}")
            elif result:
                self.channels[channel.name] = channel
                logger.info(f"✅ Kanal {channel.name} erfolgreich initialisiert")
        
        logger.info(f"Multi-Channel System bereit mit {len(self.channels)} aktiven Kanälen")
    
    async def _init_channel(self, channel: InputChannel) -> bool:
        """Initialisiert einzelnen Kanal"""
        try:
            return await channel.initialize()
        except Exception as e:
            logger.error(f"Fehler bei Initialisierung von {channel.name}: {e}")
            return False
    
    def set_message_callback(self, callback: Callable):
        """Setzt Callback für neue Nachrichten"""
        self.message_callback = callback
    
    async def process_channel_input(self, channel_name: str, raw_input: Any) -> Optional[InputMessage]:
        """Verarbeitet Input von spezifischem Kanal"""
        if channel_name not in self.channels:
            logger.error(f"Unbekannter Kanal: {channel_name}")
            return None
        
        channel = self.channels[channel_name]
        
        try:
            # Verarbeite Input
            message = await channel.process_input(raw_input)
            
            if message:
                # Update Statistiken
                self.total_messages += 1
                self.messages_by_channel[channel_name] += 1
                self.messages_by_priority[message.priority] += 1
                
                # Füge zu Queue hinzu
                await self.message_queue.put(message)
                
                # Trigger Callback wenn gesetzt
                if self.message_callback:
                    asyncio.create_task(self.message_callback(message))
                
                logger.info(f"📨 Neue Nachricht von {channel_name} (Priorität: {message.priority})")
                
                return message
            
        except Exception as e:
            logger.error(f"Fehler bei Verarbeitung von {channel_name}: {e}")
            
        return None
    
    async def get_next_message(self, timeout: Optional[float] = None) -> Optional[InputMessage]:
        """Holt nächste Nachricht aus Queue"""
        try:
            if timeout:
                return await asyncio.wait_for(self.message_queue.get(), timeout)
            else:
                return await self.message_queue.get()
        except asyncio.TimeoutError:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt Kanal-Statistiken zurück"""
        channel_stats = {}
        for name, channel in self.channels.items():
            channel_stats[name] = {
                'active': channel.active,
                'message_count': channel.message_count
            }
        
        return {
            'total_messages': self.total_messages,
            'messages_by_channel': dict(self.messages_by_channel),
            'messages_by_priority': dict(self.messages_by_priority),
            'active_channels': len([c for c in self.channels.values() if c.active]),
            'channel_details': channel_stats,
            'queue_size': self.message_queue.qsize()
        }
    
    async def simulate_channel_input(self, channel_name: str, message: str, **kwargs):
        """Simuliert Input für Testzwecke"""
        if channel_name == 'SMS':
            raw_input = {
                'from': kwargs.get('from', '+49123456789'),
                'text': message,
                'timestamp': datetime.now().isoformat(),
                'carrier': kwargs.get('carrier', 'T-Mobile')
            }
        elif channel_name == 'Email':
            raw_input = {
                'from': kwargs.get('from', 'test@example.com'),
                'subject': kwargs.get('subject', 'Test Email'),
                'body': message,
                'timestamp': datetime.now().isoformat(),
                'attachments': kwargs.get('attachments', [])
            }
        elif channel_name == 'Phone':
            raw_input = {
                'transcript': message,
                'confidence': kwargs.get('confidence', 0.9),
                'phone_number': kwargs.get('phone_number', '+49301234567'),
                'duration_seconds': kwargs.get('duration', 45),
                'timestamp': datetime.now().isoformat()
            }
        elif channel_name == 'EmergencyButton':
            raw_input = {
                'button_id': kwargs.get('button_id', 'BTN_01'),
                'location': kwargs.get('location', 'Aufzug 1'),
                'floor': kwargs.get('floor', 3),
                'building': kwargs.get('building', 'Hauptgebäude'),
                'timestamp': datetime.now().isoformat()
            }
        else:  # DirectInput
            raw_input = {
                'text': message,
                'metadata': kwargs
            }
        
        return await self.process_channel_input(channel_name, raw_input)