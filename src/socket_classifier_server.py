# socket_classifier_server.py - Zuverlässige Socket-basierte Lösung
import socket
import json
import threading
import logging
from multy_classifyer import MultiClassifierSystem, Event

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassifierSocketServer:
    def __init__(self, host='localhost', port=8888):
        self.host = host
        self.port = port
        self.system = None
        self.server_socket = None
        self.running = False
        
        # Initialisiere Klassifikator
        logger.info("Initialisiere Multi-Classifier System...")
        self.system = MultiClassifierSystem()
        logger.info("Multi-Classifier System bereit")
    
    def start(self):
        """Startet den Socket-Server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            logger.info(f"🚀 Classifier Server gestartet auf {self.host}:{self.port}")
            logger.info("Warte auf Verbindungen...")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    logger.info(f"✅ Neue Verbindung von {address}")
                    
                    # Starte Thread für Client-Behandlung
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        logger.error(f"Socket Fehler: {e}")
                        
        except Exception as e:
            logger.error(f"Server Fehler: {e}")
        finally:
            self.stop()
    
    def handle_client(self, client_socket, address):
        """Behandelt einen Client"""
        try:
            while self.running:
                # Empfange Daten
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                # Verarbeite JSON-Nachricht
                try:
                    request = json.loads(data)
                    
                    if request.get('type') == 'classify':
                        # Event-Klassifikation
                        event_data = request['event']
                        event = Event(
                            id=event_data['id'],
                            timestamp=event_data['timestamp'],
                            channel=event_data['channel'],
                            severity=event_data['severity'],
                            raw_text=event_data['rawText'],
                            source=event_data.get('source')
                        )
                        
                        # Klassifiziere Event
                        result = self.system.process_event(event)
                        
                        # Sende Antwort
                        response = {
                            'type': 'classification_result',
                            'request_id': request.get('request_id'),
                            'result': {
                                'event': {
                                    'id': result.event.id,
                                    'timestamp': result.event.timestamp,
                                    'channel': result.event.channel,
                                    'severity': result.event.severity,
                                    'rawText': result.event.raw_text,
                                    'source': result.event.source
                                },
                                'embedding_result': {
                                    'category': result.embedding_result.category,
                                    'confidence': result.embedding_result.confidence,
                                    'classifier_type': result.embedding_result.classifier_type
                                },
                                'llm_result': {
                                    'category': result.llm_result.category,
                                    'confidence': result.llm_result.confidence,
                                    'classifier_type': result.llm_result.classifier_type
                                },
                                'rule_result': {
                                    'category': result.rule_result.category,
                                    'confidence': result.rule_result.confidence,
                                    'classifier_type': result.rule_result.classifier_type
                                },
                                'consensus': result.consensus,
                                'certainty_level': result.certainty_level,
                                'should_save_for_training': result.should_save_for_training
                            }
                        }
                        
                        client_socket.send(json.dumps(response).encode('utf-8'))
                        logger.info(f"📤 Klassifikation gesendet für Event: {event.id}")
                        
                    elif request.get('type') == 'health_check':
                        # Gesundheitscheck
                        response = {
                            'type': 'health_response',
                            'status': 'healthy',
                            'message': 'Multi-Classifier System läuft'
                        }
                        client_socket.send(json.dumps(response).encode('utf-8'))
                        
                    elif request.get('type') == 'stats':
                        # Statistiken
                        stats = self.system.get_training_data_stats()
                        response = {
                            'type': 'stats_response',
                            'stats': stats
                        }
                        client_socket.send(json.dumps(response).encode('utf-8'))
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON Decode Fehler: {e}")
                    error_response = {
                        'type': 'error',
                        'message': f'Invalid JSON: {str(e)}'
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
                except Exception as e:
                    logger.error(f"Verarbeitungsfehler: {e}")
                    error_response = {
                        'type': 'error',
                        'message': f'Processing error: {str(e)}'
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    
        except Exception as e:
            logger.error(f"Client Fehler: {e}")
        finally:
            client_socket.close()
            logger.info(f"🔌 Verbindung zu {address} geschlossen")
    
    def stop(self):
        """Stoppt den Server"""
        logger.info("🛑 Stoppe Server...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    server = ClassifierSocketServer()
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("🛑 Server gestoppt durch Benutzer")
        server.stop()