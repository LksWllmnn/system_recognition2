# input_message.py - Einheitliche Input-Nachricht
from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class InputMessage:
    """Einheitliche Input-Nachricht für alle Kanäle"""
    channel: str
    raw_content: str
    processed_content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    priority: int = 0  # 0=normal, 1=high, 2=emergency