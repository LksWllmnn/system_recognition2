# __init__.py - Input Channels Package
"""
Multi-Channel Input System für Aufzugs-Klassifikator

Dieses Package bietet verschiedene Input-Kanäle:
- SMS
- Email
- Phone (Voice)
- Emergency Button
- Direct Input
"""

from .input_message import InputMessage
from .base_input_channel import InputChannel
from .sms_channel import SMSChannel
from .email_channel import EmailChannel
from .phone_channel import PhoneChannel
from .emergency_channel import EmergencyButtonChannel
from .direct_channel import DirectInputChannel
from .multi_channel_handler import MultiChannelInputHandler

__all__ = [
    'InputMessage',
    'InputChannel',
    'SMSChannel',
    'EmailChannel',
    'PhoneChannel',
    'EmergencyButtonChannel',
    'DirectInputChannel',
    'MultiChannelInputHandler'
]