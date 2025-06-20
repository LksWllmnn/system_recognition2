# demo_input_channels.py - Demo für Multi-Channel System
import asyncio
import json
import logging
from .multi_channel_handler import MultiChannelInputHandler

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def demo_multi_channel():
    """Demonstriert Multi-Channel System"""
    handler = MultiChannelInputHandler()
    
    # Initialisiere Kanäle
    await handler.initialize_all_channels()
    
    # Simuliere verschiedene Kanäle
    print("\n🔄 Teste Multi-Channel Input System...")
    
    # SMS
    await handler.simulate_channel_input(
        'SMS',
        'Aufzug in Stock 3 macht komische Geräusche bitte prüfen',
        from_='+49171234567'
    )
    
    # Email
    await handler.simulate_channel_input(
        'Email',
        'Der Aufzug stoppt abrupt zwischen den Stockwerken.',
        subject='DRINGEND: Aufzugsproblem',
        from_='facility@company.com'
    )
    
    # Phone
    await handler.simulate_channel_input(
        'Phone',
        'Hallo ähm der Aufzug im Hauptgebäude funktioniert nicht mehr richtig',
        confidence=0.85,
        duration=25
    )
    
    # Emergency Button
    await handler.simulate_channel_input(
        'EmergencyButton',
        '',  # Wird ignoriert
        button_id='BTN_03_05',
        location='Aufzug 3, Stock 5'
    )
    
    # Zeige Statistiken
    print("\n📊 Kanal-Statistiken:")
    stats = handler.get_stats()
    print(json.dumps(stats, indent=2, default=str))
    
    # Verarbeite Nachrichten aus Queue
    print("\n📬 Verarbeite Nachrichten aus Queue:")
    while not handler.message_queue.empty():
        msg = await handler.get_next_message()
        if msg:
            print(f"\n📨 Kanal: {msg.channel}")
            print(f"   Priorität: {msg.priority}")
            print(f"   Nachricht: {msg.processed_content}")
            print(f"   Metadata: {msg.metadata}")


if __name__ == "__main__":
    asyncio.run(demo_multi_channel())