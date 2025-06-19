# ollama_classifier.py - Ollama LLM Aufzugs-Klassifikator
import json
import logging
import hashlib
import asyncio
from typing import Dict, Optional
from base_classifier import BaseClassifier
from event import Event, ClassificationResult

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests nicht verfügbar - Ollama Klassifikator deaktiviert")

logger = logging.getLogger(__name__)

class OllamaLLMClassifier(BaseClassifier):
    """LLM-basierter Aufzugs-Klassifikator mit Ollama"""
    
    def __init__(self, model_name: str = "llama3.1", ollama_url: str = "http://localhost:11434"):
        super().__init__("OllamaLLM")
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.cache = {}  # Einfacher Cache für wiederholte Anfragen
        self.available = False
    
    async def initialize(self) -> bool:
        """Initialisiert und testet Ollama-Verbindung"""
        if not REQUESTS_AVAILABLE:
            logger.warning("Ollama Klassifikator übersprungen - requests nicht verfügbar")
            return False
        
        logger.info(f"Initialisiere Ollama Aufzugs-LLM-Klassifikator mit {self.model_name}...")
        
        try:
            # Teste Ollama-Verbindung
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama-Verbindung erfolgreich")
                
                # Prüfe ob Modell verfügbar ist
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if any(self.model_name in name for name in model_names):
                    logger.info(f"✅ Modell {self.model_name} verfügbar")
                    self.available = True
                    logger.info("Ollama Aufzugs-LLM-Klassifikator bereit")
                    return True
                else:
                    logger.warning(f"Modell {self.model_name} nicht gefunden. Verfügbare Modelle: {model_names}")
                    return False
            else:
                logger.warning(f"Ollama nicht erreichbar: Status {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Ollama-Verbindung fehlgeschlagen: {e}")
            return False
    
    def _get_cache_key(self, message: str) -> str:
        """Generiert Cache-Schlüssel für Message"""
        return hashlib.md5(message.encode()).hexdigest()
    
    async def classify(self, event: Event) -> ClassificationResult:
        """Klassifiziert Event mit LLM für Aufzugs-Komponenten"""
        if not self.available:
            # Fallback zu einfacher Klassifikation
            return ClassificationResult(
                event=event,
                categories={'unknown': 1.0},
                confidence=0.1,
                processing_time=0.0,
                classifier_name=self.name
            )
        
        # Prüfe Cache
        cache_key = self._get_cache_key(event.message)
        if cache_key in self.cache:
            logger.debug("Cache Hit für LLM-Klassifikation")
            cached_result = self.cache[cache_key]
            return ClassificationResult(
                event=event,
                categories=cached_result['categories'],
                confidence=cached_result['confidence'],
                processing_time=0.001,  # Cache-Zugriff
                classifier_name=self.name
            )
        
        try:
            # Aufzugs-spezifischer LLM-Prompt
            prompt = f"""
Analysiere die folgende Aufzugs-Fehlermeldung und ordne sie einem Aufzugsteil zu.

Aufzugsteile:
- fahrkabine: Türen, Bedienelemente, Innenraum, Sensoren, Beleuchtung, Notruf
- seil: Tragseile, Führungsseile, Seilrollen, Aufhängung, Seilüberwachung  
- aufzugsgetriebe: Motor, Getriebe, Schmierung, Bremsen, Steuerung, Antrieb

Nachricht: "{event.message}"

Gib das Ergebnis als JSON zurück mit Scores zwischen 0.0 und 1.0:
{{
  "fahrkabine": 0.0,
  "seil": 0.0,
  "aufzugsgetriebe": 0.0
}}

Antwort nur JSON:
"""
            
            # Asynce HTTP-Anfrage simulieren
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 150
                        }
                    },
                    timeout=15
                )
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '').strip()
                
                # Parse JSON-Antwort
                try:
                    # Entferne eventuell vorhandene Markdown-Formatierung
                    if llm_response.startswith('```'):
                        lines = llm_response.split('\n')
                        json_lines = [line for line in lines if not line.startswith('```')]
                        llm_response = '\n'.join(json_lines).strip()
                    
                    categories = json.loads(llm_response)
                    
                    if isinstance(categories, dict):
                        # Normalisiere Scores und stelle sicher, dass alle Kategorien vorhanden sind
                        normalized_categories = {}
                        for cat in ['fahrkabine', 'seil', 'aufzugsgetriebe']:
                            if cat in categories:
                                normalized_categories[cat] = max(0.0, min(1.0, float(categories[cat])))
                            else:
                                normalized_categories[cat] = 0.0
                        
                        best_category = max(normalized_categories, key=normalized_categories.get)
                        confidence = normalized_categories[best_category]
                        
                        # Cache Ergebnis
                        self.cache[cache_key] = {
                            'categories': normalized_categories,
                            'confidence': confidence
                        }
                        
                        return ClassificationResult(
                            event=event,
                            categories=normalized_categories,
                            confidence=confidence,
                            processing_time=0.0,
                            classifier_name=self.name
                        )
                except json.JSONDecodeError:
                    logger.warning(f"LLM antwortete kein valides JSON: {llm_response}")
            
        except Exception as e:
            logger.error(f"LLM-Klassifikation fehlgeschlagen: {e}")
        
        # Fallback
        return ClassificationResult(
            event=event,
            categories={'fahrkabine': 0.0, 'seil': 0.0, 'aufzugsgetriebe': 0.0, 'unknown': 1.0},
            confidence=0.1,
            processing_time=0.0,
            classifier_name=self.name
        )