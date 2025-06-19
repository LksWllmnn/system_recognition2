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

class SocketClassifierServer:
    """Socket-basierter Klassifikationsserver"""
    
    def __init__(self, host: str = 'localhost', port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = {}
        self.client_counter = 0
        self.request_count = 0
        self.start_time = time.time()
        
        # Multi-Klassifikator System
        self.classifier_system = MultiClassifierSystem()
        
        # Setup Logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialisiert den Server"""
        self.logger.info("Initialisiere Multi-Classifier System...")
        await self.classifier_system.initialize()
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
                    'request_id': request.get('request_id')
                }
            
            elif request_type == 'classify':
                # Erstelle Event
                event_data = request.get('event', {})
                event = Event(
                    timestamp=datetime.now(),
                    message=event_data.get('message', ''),
                    metadata=event_data.get('metadata', {})
                )
                
                # Klassifiziere
                result = await self.classifier_system.classify_event(event)
                
                return {
                    'type': 'classification_response',
                    'result': result,
                    'request_id': request.get('request_id')
                }
            
            elif request_type == 'stats':
                stats = self.classifier_system.get_system_stats()
                stats.update({
                    'server_uptime': time.time() - self.start_time,
                    'total_requests': self.request_count,
                    'active_clients': self.client_counter
                })
                
                return {
                    'type': 'stats_response',
                    'stats': stats,
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
    
    args = parser.parse_args()
    
    # Setup Logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    # Erstelle und starte Server
    server = SocketClassifierServer(args.host, args.port)
    
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