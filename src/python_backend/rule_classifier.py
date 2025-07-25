# rule_classifier.py - Regel-basierter Aufzugs-Klassifikator
import re
import logging
from typing import Dict, List, Tuple
from base_classifier import BaseClassifier
from event import Event, ClassificationResult

logger = logging.getLogger(__name__)

class EnhancedRuleBasedClassifier(BaseClassifier):
    """Erweiterte regel-basierte Aufzugs-Klassifikation"""
    
    def __init__(self):
        super().__init__("EnhancedRuleBased")
        self.rules = []
        self._setup_rules()
    
    def _setup_rules(self):
        """Definiert erweiterte Aufzugs-Klassifikationsregeln"""
        self.rules = [
        # FAHRKABINE - Tür-bezogene Probleme
        (r'\b(tür[a-zA-ZäöüÄÖÜß]*|türe[a-zA-ZäöüÄÖÜß]*)\b', 'fahrkabine', 0.9),
        (r'\btüröffnung\b', 'fahrkabine', 0.9),
        (r'\btürschließung\b', 'fahrkabine', 0.9),
        (r'\b(öffn|schließ|klemm)[a-zA-ZäöüÄÖÜß]*', 'fahrkabine', 0.8),
        
        # FAHRKABINE - Sensoren und Sicherheit
        (r'\btürsensor\b', 'fahrkabine', 0.9),
        (r'\b(sensor|lichtschranke)\b', 'fahrkabine', 0.7),
        (r'\b(notruf|notsprech[a-zA-ZäöüÄÖÜß]*)\b', 'fahrkabine', 0.8),
        (r'\bnotsprechanlage\b', 'fahrkabine', 0.9),
        
        # FAHRKABINE - Bedienelemente und Innenraum
        (r'\b(knopf|taste|bedien[a-zA-ZäöüÄÖÜß]*|panel|display)\b', 'fahrkabine', 0.8),
        (r'\bbedienfeld\b', 'fahrkabine', 0.9),
        (r'\b(kabine|fahrkabine|innenraum)\b', 'fahrkabine', 0.9),
        (r'\b(beleuchtung|lüftung|ventilation)\b', 'fahrkabine', 0.8),
        (r'\b(boden|wand|decke)\b', 'fahrkabine', 0.7),
        
        # SEIL - Grundlegende Seiltypen
        (r'\b(seil|seile)\b', 'seil', 0.95),
        (r'\b(tragseil|führungsseil|hubseil)\b', 'seil', 0.95),
        (r'\b(kabel|draht)\b', 'seil', 0.7),
        (r'\bseilführung\b', 'seil', 0.9),
        
        # SEIL - Seilkomponenten und Mechanik
        (r'\b(seil[a-zA-ZäöüÄÖÜß]*rolle|umlenkrolle|seilscheibe)\b', 'seil', 0.9),
        (r'\b(aufhäng[a-zA-ZäöüÄÖÜß]*|befestig[a-zA-ZäöüÄÖÜß]*)\b', 'seil', 0.8),
        (r'\bseilüberwachung\b', 'seil', 0.9),
        (r'\bspannung\b', 'seil', 0.8),
        
        # SEIL - Verschleiß und Schäden
        (r'\b(bruch|riss|verschleiß|dehnung)\b', 'seil', 0.9),
        (r'\b(bruch|riss|verschleiß|dehnung).*seil\b', 'seil', 0.95),
        
        # AUFZUGSGETRIEBE - Motor und Antrieb
        (r'\b(getriebe|motor|antrieb)\b', 'aufzugsgetriebe', 0.95),
        (r'\b(antriebsmotor|getriebemotor)\b', 'aufzugsgetriebe', 0.95),
        
        # AUFZUGSGETRIEBE - Schmierung und Öl
        (r'\b(öl|schmier[a-zA-ZäöüÄÖÜß]*)\b', 'aufzugsgetriebe', 0.9),
        (r'\b(schmierstoff|schmierölstand|getriebeöl)\b', 'aufzugsgetriebe', 0.95),
        
        # AUFZUGSGETRIEBE - Mechanische Komponenten
        (r'\b(lager|welle|zahnrad|kupplung)\b', 'aufzugsgetriebe', 0.85),
        (r'\b(bremse|bremsung)\b', 'aufzugsgetriebe', 0.85),
        (r'\b(drehzahl|geschwindigkeit|drehmoment)\b', 'aufzugsgetriebe', 0.8),
        
        # AUFZUGSGETRIEBE - Temperatur und Kühlung
        (r'\b(temperatur|überhitz[a-zA-ZäöüÄÖÜß]*|kühl[a-zA-ZäöüÄÖÜß]*)\b', 'aufzugsgetriebe', 0.7),
        (r'\bkühlung\b', 'aufzugsgetriebe', 0.8),
        
        # AUFZUGSGETRIEBE - Steuerung und Kontrolle
        (r'\b(steuer[a-zA-ZäöüÄÖÜß]*|kontrolle)\b', 'aufzugsgetriebe', 0.8),
        (r'\bsteuerungseinheit\b', 'aufzugsgetriebe', 0.9),
        
        # AUFZUGSGETRIEBE - Vibration und Bewegung
        (r'\b(vibrat[a-zA-ZäöüÄÖÜß]*|schwing[a-zA-ZäöüÄÖÜß]*|erschüttert[a-zA-ZäöüÄÖÜß]*)\b', 'aufzugsgetriebe', 0.8),
        
        # SPEZIELLE KOMBINATIONEN (aus ursprünglichen Regeln)
        (r'\btür.*verzög\w*', 'fahrkabine', 0.95),
        (r'\bgewicht.*erken\w*', 'fahrkabine', 0.9),
        (r'\binitialisier.*antrieb', 'aufzugsgetriebe', 0.9),
        (r'\bölstand.*\b(reduziert|niedrig|minimal|nahe)\b', 'aufzugsgetriebe', 0.95),
        
        # PROBLEM-INDIKATOREN (verstärken bestehende Kategorien)
        (r'\b(plötzlich|notfall|ausfall|stillstand)\b', None, 0.2),  # Verstärkt andere Scores
        (r'\b(wartung|überprüf|inspektion)\b', None, 0.1),
        ]
    
    async def initialize(self) -> bool:
        """Initialisiert den Klassifikator"""
        logger.info("Initialisiere Enhanced Aufzugs-Rule-Based Klassifikator...")
        logger.info("Enhanced Aufzugs-Rule-Based Klassifikator bereit")
        return True
    
    async def classify(self, event: Event) -> ClassificationResult:
        """Klassifiziert Event basierend auf Aufzugs-Regeln"""
        message = event.message.lower()
        scores = {'fahrkabine': 0.0, 'seil': 0.0, 'aufzugsgetriebe': 0.0}
        
        # Wende alle Regeln an
        for pattern, category, weight in self.rules:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                match_count = len(matches)
                
                if category is None:
                    # Verstärkt alle bestehenden Scores
                    for cat in scores:
                        if scores[cat] > 0:
                            scores[cat] += match_count * weight * 0.5
                else:
                    # Addiere zum spezifischen Score
                    match_score = match_count * weight
                    scores[category] += match_score
        
        # Normalisiere Scores (max 1.0)
        for category in scores:
            scores[category] = min(scores[category], 1.0)
        
        # Finde beste Kategorie
        if any(score > 0 for score in scores.values()):
            best_category = max(scores, key=scores.get)
            confidence = scores[best_category]
        else:
            best_category = 'unknown'
            confidence = 0.0
            scores['unknown'] = 1.0
        
        return ClassificationResult(
            event=event,
            categories=scores,
            confidence=confidence,
            processing_time=0.0,  # Wird später gesetzt
            classifier_name=self.name
        )