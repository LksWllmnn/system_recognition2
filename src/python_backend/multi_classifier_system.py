# multi_classifier_system.py - Multi-Klassifikator System
import asyncio
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from event import Event, ClassificationResult
from simple_classifier import SimpleEmbeddingClassifier
from rule_classifier import EnhancedRuleBasedClassifier
from ollama_classifier import OllamaLangChainClassifier
from tfidf_classifier import TfidfMLClassifier
from zero_shot_classifier import ZeroShotClassifier

# Import für erweiterte Funktionalität
try:
    from enhanced_multi_classifier import EnhancedMultiClassifierSystem
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Enhanced Multi-Classifier nicht verfügbar - verwende Standard-System")

logger = logging.getLogger(__name__)

class MultiClassifierSystem:
    """System das mehrere Klassifikatoren parallel ausführt"""
    
    def __init__(self):
        self.classifiers = []
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.total_events = 0
        self.total_time = 0.0
    
    async def initialize(self):
        """Initialisiert alle Klassifikatoren"""
        logger.info("Initialisiere Async Multi-Klassifikator-System...")
        
        # Erstelle Klassifikatoren
        self.classifiers = [
            SimpleEmbeddingClassifier(),
            EnhancedRuleBasedClassifier(),
            OllamaLangChainClassifier(),
            TfidfMLClassifier(training_data={
    # SEIL (20 Beispiele)
    "Seil quietscht beim Aufwärtsfahren": "seil",
    "Tragseil zeigt erste Verschleißerscheinungen": "seil",
    "Führungsseil hat einen kleinen Riss": "seil",
    "Hubseil ist überdehnt und muss getauscht werden": "seil",
    "Kabel an der Seilrolle ausgefranst": "seil",
    "Spannung im Tragseil zu niedrig": "seil",
    "Seilscheibe läuft unrund": "seil",
    "Umlenkrolle macht Geräusche": "seil",
    "Seilführung blockiert manchmal": "seil",
    "Aufhängung des Seils gelockert": "seil",
    "Befestigung am Seil überprüfen": "seil",
    "Seilüberwachung meldet Fehler": "seil",
    "Drahtbruch im Tragseil festgestellt": "seil",
    "Seile müssen komplett erneuert werden": "seil",
    "Verschleiß an mehreren Seilsträngen": "seil",
    "Dehnung des Hubseils überschritten": "seil",
    "Kabel zur Seilrolle defekt": "seil",
    "Seilspannung schwankt während der Fahrt": "seil",
    "Riss in der Seilummantelung entdeckt": "seil",
    "Bruch einzelner Drähte im Seil": "seil",
    
    # FAHRKABINE (20 Beispiele)
    "Kabinentür klemmt beim Schließen": "fahrkabine",
    "Türöffnung verzögert sich stark": "fahrkabine",
    "Türschließung funktioniert nicht ordnungsgemäß": "fahrkabine",
    "Kabine bleibt zwischen den Etagen stehen": "fahrkabine",
    "Beleuchtung in der Fahrkabine flackert": "fahrkabine",
    "Display zeigt falsche Etagenanzeige": "fahrkabine",
    "Knopf für 3. Stock reagiert nicht": "fahrkabine",
    "Taste im Bedienfeld klemmt": "fahrkabine",
    "Bedienfeld komplett ausgefallen": "fahrkabine",
    "Lüftung in der Kabine zu schwach": "fahrkabine",
    "Ventilation macht laute Geräusche": "fahrkabine",
    "Innenraum der Kabine zu dunkel": "fahrkabine",
    "Boden der Fahrkabine wackelt": "fahrkabine",
    "Wand in der Kabine beschädigt": "fahrkabine",
    "Deckenbeleuchtung defekt": "fahrkabine",
    "Türsensor erkennt Hindernisse nicht": "fahrkabine",
    "Lichtschranke an der Tür ausgefallen": "fahrkabine",
    "Notruf funktioniert nicht": "fahrkabine",
    "Notsprechanlage ohne Verbindung": "fahrkabine",
    "Panel zeigt Fehlermeldung an": "fahrkabine",
    
    # AUFZUGSGETRIEBE (20 Beispiele)
    "Motor überhitzt nach kurzer Laufzeit": "aufzugsgetriebe",
    "Getriebe macht ungewöhnliche Geräusche": "aufzugsgetriebe",
    "Antrieb startet verzögert": "aufzugsgetriebe",
    "Antriebsmotor läuft unruhig": "aufzugsgetriebe",
    "Getriebemotor vibriert stark": "aufzugsgetriebe",
    "Ölstand im Getriebe zu niedrig": "aufzugsgetriebe",
    "Schmierung des Motors unzureichend": "aufzugsgetriebe",
    "Schmierstoff muss gewechselt werden": "aufzugsgetriebe",
    "Schmierölstand kritisch niedrig": "aufzugsgetriebe",
    "Getriebeöl verschmutzt": "aufzugsgetriebe",
    "Vibration beim Anfahren spürbar": "aufzugsgetriebe",
    "Lager im Getriebe defekt": "aufzugsgetriebe",
    "Welle zeigt Verschleißspuren": "aufzugsgetriebe",
    "Zahnrad im Getriebe abgenutzt": "aufzugsgetriebe",
    "Kupplung rutscht durch": "aufzugsgetriebe",
    "Bremse greift nicht richtig": "aufzugsgetriebe",
    "Bremsung zu schwach eingestellt": "aufzugsgetriebe",
    "Drehzahl des Motors schwankt": "aufzugsgetriebe",
    "Geschwindigkeit nicht konstant": "aufzugsgetriebe",
    "Drehmoment zu gering": "aufzugsgetriebe",
    "Temperatur im Motorraum zu hoch": "aufzugsgetriebe",
    "Überhitzung der Antriebseinheit": "aufzugsgetriebe",
    "Kühlung des Motors unzureichend": "aufzugsgetriebe",
    "Steuerung reagiert nicht auf Befehle": "aufzugsgetriebe",
    "Steuerungseinheit zeigt Fehlercode": "aufzugsgetriebe"
}),
            ZeroShotClassifier()
        ]
        
        # Initialisiere alle Klassifikatoren
        initialization_tasks = []
        for classifier in self.classifiers:
            task = asyncio.create_task(classifier.initialize())
            initialization_tasks.append(task)
        
        # Warte auf Initialisierung
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # Prüfe Ergebnisse
        active_classifiers = []
        for i, (classifier, result) in enumerate(zip(self.classifiers, results)):
            if isinstance(result, Exception):
                logger.warning(f"Klassifikator {classifier.name} Initialisierung fehlgeschlagen: {result}")
            elif result:
                active_classifiers.append(classifier)
                logger.info(f"✅ {classifier.name} erfolgreich initialisiert")
            else:
                logger.warning(f"Klassifikator {classifier.name} nicht verfügbar")
        
        self.classifiers = active_classifiers
        logger.info("Async Multi-Klassifikator-System bereit")
    
    async def classify_event(self, event: Event) -> Dict[str, Any]:
        """Klassifiziert Event mit allen verfügbaren Klassifikatoren"""
        if not self.classifiers:
            logger.warning("Keine aktiven Klassifikatoren verfügbar")
            return {
                'event': event.to_dict(),
                'results': [],
                'combined_score': {'unknown': 1.0},
                'processing_time': 0.0
            }
        
        start_time = asyncio.get_event_loop().time()
        
        # Führe alle Klassifikatoren parallel aus
        classification_tasks = []
        for classifier in self.classifiers:
            task = asyncio.create_task(classifier.classify_with_timing(event))
            classification_tasks.append(task)
        
        # Warte auf alle Ergebnisse
        results = await asyncio.gather(*classification_tasks, return_exceptions=True)
        
        # Verarbeite Ergebnisse
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Klassifikator {self.classifiers[i].name} Fehler: {result}")
            else:
                valid_results.append(result)
        
        # Kombiniere Ergebnisse
        combined_scores = self._combine_results(valid_results)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        self.total_events += 1
        self.total_time += processing_time
        
        return {
            'event': event.to_dict(),
            'results': [result.to_dict() for result in valid_results],
            'combined_score': combined_scores,
            'processing_time': processing_time,
            'classifier_count': len(valid_results)
        }
    
    def _combine_results(self, results: List[ClassificationResult]) -> Dict[str, float]:
        """Kombiniert Ergebnisse mehrerer Klassifikatoren"""
        if not results:
            return {'unknown': 1.0}
        
        # Sammle alle Kategorien
        all_categories = set()
        for result in results:
            all_categories.update(result.categories.keys())
        
        # Berechne gewichtete Durchschnitte
        combined = {}
        for category in all_categories:
            scores = []
            weights = []
            
            for result in results:
                if category in result.categories:
                    scores.append(result.categories[category])
                    weights.append(result.confidence)  # Gewichte basierend auf Confidence
                else:
                    scores.append(0.0)
                    weights.append(0.1)  # Geringe Gewichtung für fehlende Kategorien
            
            # Gewichteter Durchschnitt
            if sum(weights) > 0:
                combined[category] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                combined[category] = sum(scores) / len(scores)
        
        return combined
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Gibt System-Statistiken zurück"""
        classifier_stats = {}
        for classifier in self.classifiers:
            classifier_stats[classifier.name] = classifier.get_stats()
        
        avg_time = self.total_time / max(1, self.total_events)
        
        return {
            'total_events': self.total_events,
            'total_time': self.total_time,
            'average_time': avg_time,
            'active_classifiers': len(self.classifiers),
            'classifier_stats': classifier_stats
        }