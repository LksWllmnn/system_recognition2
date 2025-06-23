# ollama_langchain_classifier.py - Ollama LLM Klassifikator mit LangChain Function Calling
import json
import logging
import hashlib
import asyncio
from typing import Dict, Optional, List

# Mock-Klassen für Standalone-Test
class Event:
    def __init__(self, message: str, timestamp=None):
        self.message = message
        self.timestamp = timestamp or asyncio.get_event_loop().time()
    
    def to_dict(self):
        return {"message": self.message, "timestamp": self.timestamp}

class ClassificationResult:
    def __init__(self, event, categories, confidence, processing_time, classifier_name):
        self.event = event
        self.categories = categories
        self.confidence = confidence
        self.processing_time = processing_time
        self.classifier_name = classifier_name
    
    def to_dict(self):
        return {
            "event": self.event.to_dict(),
            "categories": self.categories,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "classifier_name": self.classifier_name
        }

class BaseClassifier:
    def __init__(self, name):
        self.name = name
        self.classification_count = 0
        self.total_time = 0.0
    
    async def classify_with_timing(self, event):
        start_time = asyncio.get_event_loop().time()
        result = await self.classify(event)
        end_time = asyncio.get_event_loop().time()
        result.processing_time = end_time - start_time
        self.classification_count += 1
        self.total_time += result.processing_time
        return result
    
    def get_stats(self):
        avg_time = self.total_time / max(1, self.classification_count)
        return {
            "count": self.classification_count,
            "total_time": self.total_time,
            "average_time": avg_time
        }

try:
    from langchain_community.llms import Ollama
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain.schema import AgentAction, AgentFinish
    from langchain.prompts import PromptTemplate
    from langchain.agents.format_scratchpad import format_log_to_str
    from langchain.agents.output_parsers import ReActSingleInputOutputParser
    from langchain.tools.render import render_text_description
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain nicht verfügbar - Ollama LangChain Klassifikator deaktiviert")

logger = logging.getLogger(__name__)

class OllamaLangChainClassifier(BaseClassifier):
    """LLM-basierter Aufzugs-Klassifikator mit LangChain Function Calling"""
    
    def __init__(self, model_name: str = "llama3.1", ollama_url: str = "http://localhost:11434"):
        super().__init__("OllamaLangChain")
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.cache = {}
        self.available = False
        self.llm = None
        self.agent = None
        self.tools = []
        
        # Ergebnis-Storage für Function Calls
        self.classification_result = None
    
    def _classify_as_fahrkabine(self, confidence: str = "0.8") -> str:
        """Klassifiziert als Fahrkabine-Problem"""
        try:
            conf = float(confidence)
            self.classification_result = {
                'fahrkabine': conf,
                'seil': 0.0,
                'aufzugsgetriebe': 0.0
            }
            return f"Klassifiziert als Fahrkabine-Problem mit Konfidenz {conf}"
        except:
            self.classification_result = {'fahrkabine': 0.8, 'seil': 0.0, 'aufzugsgetriebe': 0.0}
            return "Klassifiziert als Fahrkabine-Problem mit Konfidenz 0.8"
    
    def _classify_as_seil(self, confidence: str = "0.8") -> str:
        """Klassifiziert als Seil-Problem"""
        try:
            conf = float(confidence)
            self.classification_result = {
                'fahrkabine': 0.0,
                'seil': conf,
                'aufzugsgetriebe': 0.0
            }
            return f"Klassifiziert als Seil-Problem mit Konfidenz {conf}"
        except:
            self.classification_result = {'fahrkabine': 0.0, 'seil': 0.8, 'aufzugsgetriebe': 0.0}
            return "Klassifiziert als Seil-Problem mit Konfidenz 0.8"
    
    def _classify_as_aufzugsgetriebe(self, confidence: str = "0.8") -> str:
        """Klassifiziert als Aufzugsgetriebe-Problem"""
        try:
            conf = float(confidence)
            self.classification_result = {
                'fahrkabine': 0.0,
                'seil': 0.0,
                'aufzugsgetriebe': conf
            }
            return f"Klassifiziert als Aufzugsgetriebe-Problem mit Konfidenz {conf}"
        except:
            self.classification_result = {'fahrkabine': 0.0, 'seil': 0.0, 'aufzugsgetriebe': 0.8}
            return "Klassifiziert als Aufzugsgetriebe-Problem mit Konfidenz 0.8"
    
    def _classify_as_unknown(self, confidence: str = "0.5") -> str:
        """Klassifiziert als unbekanntes Problem"""
        try:
            conf = float(confidence)
            self.classification_result = {
                'fahrkabine': 0.0,
                'seil': 0.0,
                'aufzugsgetriebe': 0.0,
                'unknown': conf
            }
            return f"Kann nicht klassifiziert werden mit Konfidenz {conf}"
        except:
            self.classification_result = {'fahrkabine': 0.0, 'seil': 0.0, 'aufzugsgetriebe': 0.0, 'unknown': 0.5}
            return "Kann nicht klassifiziert werden mit Konfidenz 0.5"
    
    def _setup_tools(self):
        """Erstellt die verfügbaren Tools für das LLM"""
        self.tools = [
            Tool(
                name="classify_fahrkabine",
                func=self._classify_as_fahrkabine,
                description="Verwende diese Funktion wenn das Problem mit der Fahrkabine zusammenhängt (Türen, Beleuchtung, Innenraum, Bedienfeld, etc.). Parameter: confidence (0.0-1.0)"
            ),
            Tool(
                name="classify_seil",
                func=self._classify_as_seil,
                description="Verwende diese Funktion wenn das Problem mit dem Seil zusammenhängt (Seile, Kabel, Aufhängung, Seilführung, etc.). Parameter: confidence (0.0-1.0)"
            ),
            Tool(
                name="classify_aufzugsgetriebe",
                func=self._classify_as_aufzugsgetriebe,
                description="Verwende diese Funktion wenn das Problem mit dem Aufzugsgetriebe zusammenhängt (Motor, Getriebe, Antrieb, Steuerung, etc.). Parameter: confidence (0.0-1.0)"
            ),
            Tool(
                name="classify_unknown",
                func=self._classify_as_unknown,
                description="Verwende diese Funktion wenn das Problem keiner der drei Kategorien zugeordnet werden kann. Parameter: confidence (0.0-1.0)"
            )
        ]
    
    async def initialize(self) -> bool:
        """Initialisiert LangChain mit Ollama"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain Klassifikator übersprungen - LangChain nicht verfügbar")
            return False
        
        logger.info(f"Initialisiere LangChain Ollama Klassifikator mit {self.model_name}...")
        
        try:
            # Initialisiere Ollama LLM
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.ollama_url,
                temperature=0.1,
                num_predict=200
            )
            
            # Teste Verbindung mit einfacher Anfrage
            test_response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.llm.invoke("Sage nur 'Test erfolgreich'")
            )
            
            if "test erfolgreich" in test_response.lower():
                logger.info("✅ LangChain Ollama-Verbindung erfolgreich")
                
                # Setup Tools
                self._setup_tools()
                
                # Erstelle benutzerdefinierten Prompt
                template = """Du bist ein Experte für Aufzugs-Diagnose. Analysiere die folgende Aufzugs-Meldung und klassifiziere sie in eine der drei Kategorien:

1. **fahrkabine** - Probleme mit der Fahrkabine (Türen, Beleuchtung, Innenraum, Bedienfeld, Anzeigen)
2. **seil** - Probleme mit Seilen und Kabeln (Aufhängung, Seilführung, Kabel, Drahtseile)
3. **aufzugsgetriebe** - Probleme mit Antrieb und Getriebe (Motor, Getriebe, Steuerung, Elektronik)

Aufzugs-Meldung: "{input}"

Du hast Zugriff auf diese Tools:
{tools}

Verwende das folgende Format:

Thought: Ich muss die Meldung analysieren und die passende Kategorie bestimmen
Action: [eine der verfügbaren Funktionen]
Action Input: [Konfidenz-Wert zwischen 0.0 und 1.0]
Observation: [Ergebnis der Funktion]
Thought: Ich weiß jetzt die Antwort
Final Answer: [Zusammenfassung der Klassifikation]

Beginne!

Thought: {agent_scratchpad}"""

                prompt = PromptTemplate.from_template(template)
                
                # Erstelle Agent manually für bessere Kontrolle
                self.agent_prompt = prompt.partial(
                    tools=render_text_description(self.tools),
                )
                
                self.available = True
                logger.info("LangChain Ollama Klassifikator bereit")
                return True
            else:
                logger.warning("LangChain Ollama Test fehlgeschlagen")
                return False
                
        except Exception as e:
            logger.warning(f"LangChain Ollama-Verbindung fehlgeschlagen: {e}")
            return False
    
    def _get_cache_key(self, message: str) -> str:
        """Generiert Cache-Schlüssel für Message"""
        return hashlib.md5(message.encode()).hexdigest()
    
    async def _run_agent(self, message: str) -> Dict[str, float]:
        """Führt den Agent aus und gibt Klassifikationsergebnis zurück"""
        # Reset classification result
        self.classification_result = None
        
        # Erstelle den vollständigen Prompt
        full_prompt = self.agent_prompt.format(
            input=message,
            agent_scratchpad=""
        )
        
        try:
            # Führe LLM aus
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(full_prompt)
            )
            
            # Parse die Antwort und führe Actions aus
            lines = response.strip().split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('Action:'):
                    action_name = line.replace('Action:', '').strip()
                    
                    # Suche nach Action Input in der nächsten Zeile
                    action_input = "0.8"  # Default
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('Action Input:'):
                            action_input = next_line.replace('Action Input:', '').strip()
                    
                    # Führe entsprechende Funktion aus
                    if 'fahrkabine' in action_name.lower():
                        self._classify_as_fahrkabine(action_input)
                        break
                    elif 'seil' in action_name.lower():
                        self._classify_as_seil(action_input)
                        break
                    elif 'getriebe' in action_name.lower():
                        self._classify_as_aufzugsgetriebe(action_input)
                        break
                    elif 'unknown' in action_name.lower():
                        self._classify_as_unknown(action_input)
                        break
            
            # Fallback falls keine Action erkannt wurde
            if self.classification_result is None:
                # Versuche Schlüsselwörter zu erkennen
                message_lower = message.lower()
                if any(word in message_lower for word in ['tür', 'kabine', 'licht', 'bedien', 'anzeige']):
                    self._classify_as_fahrkabine("0.7")
                elif any(word in message_lower for word in ['seil', 'kabel', 'draht', 'aufhäng']):
                    self._classify_as_seil("0.7")
                elif any(word in message_lower for word in ['motor', 'getriebe', 'antrieb', 'steuer']):
                    self._classify_as_aufzugsgetriebe("0.7")
                else:
                    self._classify_as_unknown("0.5")
            
            return self.classification_result
            
        except Exception as e:
            logger.error(f"Agent-Ausführung fehlgeschlagen: {e}")
            return {'fahrkabine': 0.0, 'seil': 0.0, 'aufzugsgetriebe': 0.0, 'unknown': 1.0}
    
    async def classify(self, event: Event) -> ClassificationResult:
        """Klassifiziert Event mit LangChain Function Calling"""
        if not self.available:
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
            logger.debug("Cache Hit für LangChain-Klassifikation")
            cached_result = self.cache[cache_key]
            return ClassificationResult(
                event=event,
                categories=cached_result['categories'],
                confidence=cached_result['confidence'],
                processing_time=0.001,
                classifier_name=self.name
            )
        
        try:
            # Führe Agent aus
            categories = await self._run_agent(event.message)
            
            # Berechne Confidence
            confidence = max(categories.values()) if categories else 0.1
            
            # Cache Ergebnis
            self.cache[cache_key] = {
                'categories': categories,
                'confidence': confidence
            }
            
            return ClassificationResult(
                event=event,
                categories=categories,
                confidence=confidence,
                processing_time=0.0,
                classifier_name=self.name
            )
            
        except Exception as e:
            logger.error(f"LangChain-Klassifikation fehlgeschlagen: {e}")
            return ClassificationResult(
                event=event,
                categories={'fahrkabine': 0.0, 'seil': 0.0, 'aufzugsgetriebe': 0.0, 'unknown': 1.0},
                confidence=0.1,
                processing_time=0.0,
                classifier_name=self.name
            )


# Standalone Test-Main
async def main():
    """Test-Main für den OllamaLangChainClassifier"""
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test-Prompts für verschiedene Kategorien
    test_messages = [
        # Fahrkabine-Tests
        "Die Kabinentür klemmt beim Schließen",
        "Beleuchtung in der Fahrkabine flackert",
        "Bedienfeld reagiert nicht auf Knopfdruck",
        "Türsensor funktioniert nicht richtig",
        "Anzeige zeigt falsche Etage an",
        
        # Seil-Tests
        "Seil quietscht beim Aufwärtsfahren",
        "Tragseil zeigt Verschleißspuren",
        "Kabel hängt durch",
        "Seilführung ist beschädigt",
        "Drahtseil hat einzelne gebrochene Drähte",
        
        # Aufzugsgetriebe-Tests
        "Motor überhitzt nach kurzer Laufzeit",
        "Getriebe macht ungewöhnliche Geräusche",
        "Antrieb ruckelt beim Anfahren",
        "Steuerung reagiert verzögert",
        "Frequenzumrichter zeigt Fehlermeldung",
        
        # Unklare Tests
        "Aufzug riecht komisch",
        "Seltsame Vibrationen im Schacht",
        "Ungewöhnliche Geräusche"
    ]
    
    print("🚀 Starte OllamaLangChainClassifier Test...\n")
    
    # Erstelle und initialisiere Klassifikator
    classifier = OllamaLangChainClassifier(
        model_name="llama3.1",  # Ändere hier das Modell falls nötig
        ollama_url="http://localhost:11434"
    )
    
    print("📡 Initialisiere Klassifikator...")
    success = await classifier.initialize()
    
    if not success:
        print("❌ Klassifikator-Initialisierung fehlgeschlagen!")
        print("Stelle sicher, dass:")
        print("- Ollama läuft (ollama serve)")
        print("- Das Modell verfügbar ist (ollama list)")
        print("- LangChain installiert ist (pip install langchain langchain-community)")
        return
    
    print("✅ Klassifikator erfolgreich initialisiert!\n")
    print("=" * 80)
    
    # Teste alle Nachrichten
    results = []
    for i, message in enumerate(test_messages, 1):
        print(f"\n🔍 Test {i}/{len(test_messages)}: '{message}'")
        print("-" * 60)
        
        # Erstelle Event
        event = Event(message)
        
        # Klassifiziere
        try:
            result = await classifier.classify_with_timing(event)
            results.append(result)
            
            # Zeige Ergebnis
            print(f"📊 Kategorien: {result.categories}")
            print(f"🎯 Konfidenz: {result.confidence:.2f}")
            print(f"⏱️  Zeit: {result.processing_time:.3f}s")
            
            # Zeige beste Kategorie
            best_category = max(result.categories.items(), key=lambda x: x[1])
            print(f"🏆 Beste Kategorie: {best_category[0]} ({best_category[1]:.2f})")
            
        except Exception as e:
            print(f"❌ Fehler bei Klassifikation: {e}")
            continue
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("📈 ZUSAMMENFASSUNG")
    print("=" * 80)
    
    if results:
        # Statistiken
        stats = classifier.get_stats()
        print(f"🔢 Anzahl Tests: {stats['count']}")
        print(f"⏱️  Durchschnittliche Zeit: {stats['average_time']:.3f}s")
        print(f"🕒 Gesamtzeit: {stats['total_time']:.2f}s")
        
        # Kategorieverteilung
        category_counts = {}
        for result in results:
            best_cat = max(result.categories.items(), key=lambda x: x[1])[0]
            category_counts[best_cat] = category_counts.get(best_cat, 0) + 1
        
        print(f"\n📊 Kategorieverteilung:")
        for category, count in category_counts.items():
            print(f"   {category}: {count} Tests")
        
        # Durchschnittliche Konfidenz
        avg_confidence = sum(r.confidence for r in results) / len(results)
        print(f"\n🎯 Durchschnittliche Konfidenz: {avg_confidence:.2f}")
        
    print(f"\n✅ Test abgeschlossen!")

# Interaktiver Test
async def interactive_test():
    """Interaktiver Test-Modus"""
    logging.basicConfig(level=logging.WARNING)  # Weniger Logs
    
    print("🤖 Interaktiver OllamaLangChainClassifier Test")
    print("=" * 50)
    
    classifier = OllamaLangChainClassifier()
    
    print("📡 Initialisiere Klassifikator...")
    if not await classifier.initialize():
        print("❌ Initialisierung fehlgeschlagen!")
        return
    
    print("✅ Bereit! Gib Aufzugs-Meldungen ein (oder 'quit' zum Beenden)\n")
    
    while True:
        try:
            message = input("💬 Meldung: ").strip()
            
            if message.lower() in ['quit', 'exit', 'q']:
                break
            
            if not message:
                continue
            
            # Klassifiziere
            event = Event(message)
            result = await classifier.classify_with_timing(event)
            
            # Zeige Ergebnis
            print(f"📊 {result.categories}")
            best_cat = max(result.categories.items(), key=lambda x: x[1])
            print(f"🏆 {best_cat[0]} ({best_cat[1]:.2f}) in {result.processing_time:.2f}s\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Fehler: {e}\n")
    
    print("👋 Auf Wiedersehen!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_test())
    else:
        asyncio.run(main())