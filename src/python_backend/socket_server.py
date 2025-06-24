# socket_server.py - Sauberer Socket Server
# -*- coding: utf-8 -*-
import sys
import os

# Setze UTF-8 Encoding für Windows-Kompatibilität
if os.name == 'nt':  # Windows
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

import socket
import json
import threading
import logging
import struct
import time
import argparse
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime

# Importiere unsere Module
from event import Event
from multi_classifier_system import MultiClassifierSystem

try:
    from enhanced_multi_classifier import EnhancedMultiClassifierSystem
    from input_channels.multi_channel_handler import MultiChannelInputHandler
    from input_channels.input_message import InputMessage
    from enhanced_logging import ClassificationLogger, VisualizationHelper
    ENHANCED_MODE = True
except ImportError:
    from multi_classifier_system import MultiClassifierSystem
    ENHANCED_MODE = False
    print("Erweiterte Module nicht verfügbar - verwende Standard-Modus")

class SocketClassifierServer:
    """Socket-basierter Klassifikationsserver"""
    
    def __init__(self, host: str = 'localhost', port: int = 8888, use_enhanced: bool = True):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = {}
        self.client_counter = 0
        self.request_count = 0
        self.start_time = time.time()

        # Setup Logging zuerst!
        self.logger = logging.getLogger(__name__)

        # Wähle Klassifikator-System
        if use_enhanced and ENHANCED_MODE:
            self.classifier_system = EnhancedMultiClassifierSystem()
            self.input_handler = MultiChannelInputHandler()
            self.classification_logger = ClassificationLogger()
            self.enhanced_mode = True
            self.classification_logger.enable_performance_tracking()
            self.logger.info("Verwende Enhanced Multi-Classifier System")
        else:
            self.classifier_system = MultiClassifierSystem()
            self.input_handler = None
            self.classification_logger = None
            self.enhanced_mode = False
            self.logger.info("Verwende Standard Multi-Classifier System")

        # Setup Logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialisiert den Server"""
        self.logger.info("Initialisiere Multi-Classifier System...")
        await self.classifier_system.initialize()
        # Initialisiere Input-Handler wenn im Enhanced Mode
        if self.enhanced_mode and self.input_handler:
            await self.input_handler.initialize_all_channels()
            self.logger.info("Input-Kanäle initialisiert")

        self.logger.info("Multi-Classifier System bereit")
    
    def start_server(self):
        """Startet den Socket-Server"""
        try:
            # Erstelle Socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Binde an Adresse
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            self.running = True
            self.logger.info(f"🚀 Classifier Server gestartet auf {self.host}:{self.port}")
            self.logger.info("Warte auf Verbindungen...")
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    self.client_counter += 1
                    
                    self.logger.info(f"✅ Neue Verbindung von {address} (Client #{self.client_counter})")
                    
                    # Starte Client-Handler in eigenem Thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Socket-Fehler: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Server-Fehler: {e}")
        finally:
            self.stop_server()
    
    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Behandelt einen Client"""
        client_id = f"{address[0]}:{address[1]}"
        
        try:
            client_socket.settimeout(30.0)
            
            while self.running:
                # Empfange Nachricht
                request = self.receive_message(client_socket)
                if not request:
                    break
                
                # Verarbeite Request
                response = asyncio.run(self.process_request(request, client_id))
                
                # Sende Antwort
                if not self.send_message(client_socket, response):
                    break
                    
        except socket.timeout:
            self.logger.debug(f"Client {client_id} Timeout")
        except Exception as e:
            self.logger.error(f"Client {client_id} Fehler: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            self.logger.info(f"🔌 Verbindung zu {client_id} geschlossen")
    
    async def process_request(self, request: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Verarbeitet eine Client-Anfrage"""
        self.request_count += 1
        request_type = request.get('type', 'unknown')

        self.logger.debug(f"Verarbeite {request_type} von {client_id}")

        try:
            if request_type == 'health_check':
                return {
                    'type': 'health_response',
                    'status': 'ok',
                    'uptime_seconds': time.time() - self.start_time,
                    'total_requests': self.request_count,
                    'active_clients': self.client_counter,
                    'enhanced_mode': self.enhanced_mode,
                    'request_id': request.get('request_id')
                }

            elif request_type == 'classify':
                # Erstelle Event
                event_data = request.get('event', {})
                
                # Extrahiere Ground Truth aus Metadata wenn vorhanden
                metadata = event_data.get('metadata', {})
                true_category = metadata.pop('expected_category', None)  # NEU!
            
                # Prüfe ob Channel-Information vorhanden
                channel = event_data.get('channel')
                if channel and self.enhanced_mode and self.input_handler:
                    # Verarbeite durch Input-Handler
                    input_message = await self.input_handler.process_channel_input(
                        channel,
                        event_data
                    )
            
                    if input_message:
                        event = Event(
                            timestamp=input_message.timestamp,
                            message=input_message.processed_content,
                            metadata={
                                'channel': input_message.channel,
                                'priority': input_message.priority,
                                **input_message.metadata
                            }
                        )
                    else:
                        # Fallback zu direkter Verarbeitung
                        event = Event(
                            timestamp=datetime.now(),
                            message=event_data.get('message', ''),
                            metadata=metadata
                        )
                else:
                    # Standard Event-Erstellung
                    event = Event(
                        timestamp=datetime.now(),
                        message=event_data.get('message', ''),
                        metadata=metadata
                    )
            
                # Klassifiziere mit Ground Truth
                result = await self.classifier_system.classify_event(event, true_category=true_category)  # NEU!
            
                # Logge wenn Enhanced Logger verfügbar
                if self.enhanced_mode and self.classification_logger:
                    self.classification_logger.log_classification_result(result)
            
                return {
                    'type': 'classification_response',
                    'result': result,
                    'request_id': request.get('request_id')
                }

            elif request_type == 'stats':
                if self.enhanced_mode and hasattr(self.classifier_system, 'get_enhanced_stats'):
                    stats = self.classifier_system.get_enhanced_stats()
                else:
                    stats = self.classifier_system.get_system_stats()

                stats.update({
                    'server_uptime': time.time() - self.start_time,
                    'total_requests': self.request_count,
                    'active_clients': self.client_counter,
                    'enhanced_mode': self.enhanced_mode
                })

                # Füge Input-Handler Stats hinzu wenn verfügbar
                if self.enhanced_mode and self.input_handler:
                    stats['input_channels'] = self.input_handler.get_stats()
                # Füge Performance-Report zu stats hinzu:
                if self.enhanced_mode and self.classification_logger:
                    stats['performance_report'] = self.classification_logger.generate_performance_report()

                return {
                    'type': 'stats_response',
                    'stats': stats,
                    'request_id': request.get('request_id')
                }

            elif request_type == 'channel_input' and self.enhanced_mode:
                # Direkte Channel-Input Verarbeitung
                channel = request.get('channel')
                data = request.get('data')

                if self.input_handler:
                    input_message = await self.input_handler.process_channel_input(channel, data)

                    if input_message:
                        # Erstelle Event und klassifiziere
                        event = Event(
                            timestamp=input_message.timestamp,
                            message=input_message.processed_content,
                            metadata={
                                'channel': input_message.channel,
                                'priority': input_message.priority,
                                **input_message.metadata
                            }
                        )

                        result = await self.classifier_system.classify_event(event)

                        if self.classification_logger:
                            self.classification_logger.log_classification_result(result)

                        return {
                            'type': 'channel_response',
                            'input_message': {
                                'channel': input_message.channel,
                                'processed_content': input_message.processed_content,
                                'priority': input_message.priority
                            },
                            'classification': result,
                            'request_id': request.get('request_id')
                        }

                return {
                    'type': 'error',
                    'message': 'Channel input processing failed',
                    'request_id': request.get('request_id')
                }

            else:
                return {
                    'type': 'error',
                    'message': f'Unbekannter Request-Typ: {request_type}',
                    'request_id': request.get('request_id')
                }

        except Exception as e:
            self.logger.error(f"Fehler bei Request-Verarbeitung: {e}")
            return {
                'type': 'error',
                'message': f'Processing error: {str(e)}',
                'request_id': request.get('request_id')
            }
    
    def send_message(self, client_socket: socket.socket, data: Dict) -> bool:
        """Sendet Nachricht mit Message Framing"""
        try:
            # Serialisiere JSON
            message = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            
            # Sende Länge als 4-Byte Big-Endian Integer
            length = struct.pack('!I', len(message))
            
            # Sende Länge + Nachricht
            client_socket.sendall(length + message)
            
            self.logger.debug(f"Nachricht gesendet: {len(message)} bytes, Type: {data.get('type')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Senden: {e}")
            return False
    
    def receive_message(self, client_socket: socket.socket) -> Optional[Dict]:
        """Empfängt Nachricht mit Message Framing"""
        try:
            # Empfange Länge (4 Bytes)
            length_data = self.receive_exact(client_socket, 4)
            if not length_data:
                return None
            
            # Dekodiere Länge
            message_length = struct.unpack('!I', length_data)[0]
            
            # Sicherheitscheck
            if message_length > 1024 * 1024:  # Max 1MB
                self.logger.error(f"Nachricht zu groß: {message_length}")
                return None
            
            # Empfange Nachricht
            message_data = self.receive_exact(client_socket, message_length)
            if not message_data:
                return None
            
            # Dekodiere JSON
            request = json.loads(message_data.decode('utf-8'))
            self.logger.debug(f"Nachricht empfangen: {message_length} bytes, Type: {request.get('type')}")
            return request
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON Decode Fehler: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Fehler beim Empfangen: {e}")
            return None
    
    def receive_exact(self, client_socket: socket.socket, num_bytes: int) -> Optional[bytes]:
        """Empfängt exakt num_bytes"""
        data = b''
        while len(data) < num_bytes:
            try:
                chunk = client_socket.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.error:
                return None
        return data
    
    def stop_server(self):
        """Stoppt den Server"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.logger.info("Server gestoppt")


def main():
    """Hauptfunktion"""
    parser = argparse.ArgumentParser(description='Socket Classifier Server')
    parser.add_argument('--host', default='localhost', help='Server Host')
    parser.add_argument('--port', type=int, default=8888, help='Server Port')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--enhanced', action='store_true', help='Verwende Enhanced Mode mit Multi-Channel Support')
    
    args = parser.parse_args()
    
    # Setup Logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Erstelle und starte Server
    server = SocketClassifierServer(args.host, args.port, use_enhanced=args.enhanced)
    
    try:
        # Initialisiere Server (async)
        asyncio.run(server.initialize())
        
        # Starte Server (sync)
        server.start_server()
        
    except KeyboardInterrupt:
        print("\n🛑 Server gestoppt durch Benutzer")
    except Exception as e:
        print(f"❌ Server Fehler: {e}")
    finally:
        server.stop_server()


if __name__ == "__main__":
    main()