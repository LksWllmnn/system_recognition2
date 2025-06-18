# multy_classifyer_simple.py - Alternative ohne sentence-transformers
import numpy as np
import re
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from collections import Counter

print(" Simple Multi-Classifier Starting...")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    classifier_type: str

@dataclass
class Event:
    id: str
    timestamp: float
    channel: str
    severity: str
    raw_text: str
    source: Optional[str] = None

@dataclass
class ProcessedEvent:
    event: Event
    embedding_result: ClassificationResult
    llm_result: ClassificationResult
    rule_result: ClassificationResult
    consensus: str
    certainty_level: str
    should_save_for_training: bool

class SimpleEmbeddingClassifier:
    """Einfacher Embedding-Klassifikator ohne sentence-transformers"""
    
    def __init__(self):
        logger.info("Initialisiere Simple Embedding-Klassifikator...")
        
        # Keyword-basierte "Embeddings" (TF-IDF ähnlich)
        self.category_keywords = {
            "fahrstuhl_kabine": {
                "kabine": 5, "tür": 4, "türen": 4, "notruf": 5, "fahrgast": 4,
                "innenraum": 3, "licht": 2, "gewicht": 3, "überlast": 4,
                "steckt": 4, "eingeschlossen": 5, "notfall": 4, "person": 3,
                "passagier": 3, "festsitzen": 4, "blockiert": 3
            },
            "fahrstuhl_seil": {
                "seil": 5, "seile": 5, "spannung": 4, "umlenkrolle": 5, 
                "aufhängung": 4, "tragsystem": 5, "materialermüdung": 5,
                "befestigung": 3, "führung": 2, "tragseil": 5, "kabel": 3,
                "draht": 2, "bruch": 4, "riss": 4, "verschleiß": 3
            },
            "fahrstuhl_getriebe": {
                "getriebe": 5, "motor": 4, "antrieb": 4, "öl": 3,
                "maschinenraum": 4, "hydraulik": 4, "geräusch": 3,
                "überhitzt": 4, "druck": 2, "pumpe": 3, "ventilator": 2,
                "kühlung": 2, "temperatur": 2, "schmierung": 3, "lager": 3
            }
        }
        
        logger.info("Simple Embedding-Klassifikator bereit")
    
    def classify(self, text: str, threshold: float = 0.3) -> ClassificationResult:
        """Klassifiziert Text basierend auf Keyword-Scoring"""
        text_lower = text.lower()
        
        scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0
            word_count = 0
            
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    score += weight
                    word_count += 1
            
            # Normalisiere Score
            if word_count > 0:
                scores[category] = score / (len(text.split()) + 1)  # +1 gegen Division durch 0
            else:
                scores[category] = 0
        
        if not scores or max(scores.values()) < threshold:
            return ClassificationResult(
                category="kein_fokus",
                confidence=0.8,
                classifier_type="simple_embedding"
            )
        
        best_category = max(scores, key=scores.get)
        confidence = min(0.95, scores[best_category] * 2)  # Skaliere auf 0-0.95
        
        return ClassificationResult(
            category=best_category,
            confidence=round(confidence, 3),
            classifier_type="simple_embedding"
        )

class AdvancedLLMClassifier:
    """Erweiterte LLM-Simulation ohne externe Dependencies"""
    
    def __init__(self):
        logger.info("Initialisiere Advanced LLM-Klassifikator...")
        
        # Erweiterte Regel-Sets mit Kontext
        self.classification_rules = {
            "fahrstuhl_kabine": [
                # Direkte Kabinen-Begriffe
                (r'\bkabine\b', 4),
                (r'\btür(?:en?)?\b', 3),
                (r'\bnotruf\b', 5),
                (r'\bfahrgast\b', 3),
                (r'\binnenraum\b', 3),
                
                # Kontextuelle Begriffe
                (r'\b(?:überlast|gewicht.*überschritten)\b', 4),
                (r'\b(?:steckt.*fest|blockiert.*kabine)\b', 5),
                (r'\b(?:licht.*kabine|kabinen.*licht)\b', 3),
                (r'\b(?:person.*eingeschlossen|eingeschlossen.*person)\b', 5),
                
                # Negative Indikatoren (reduzieren Score)
                (r'\b(?:maschinenraum|getriebe|motor)\b', -2),
                (r'\b(?:seil|spannung|aufhängung)\b', -2),
            ],
            
            "fahrstuhl_seil": [
                # Direkte Seil-Begriffe  
                (r'\bseil(?:e)?\b', 5),
                (r'\bspannung\b', 4),
                (r'\bumlenkrolle\b', 5),
                (r'\baufhängung\b', 4),
                (r'\btragsystem\b', 5),
                
                # Kontextuelle Begriffe
                (r'\b(?:materialermüdung|verschleiß.*seil)\b', 5),
                (r'\b(?:seil.*gerissen|bruch.*seil)\b', 5),
                (r'\b(?:befestigung.*locker|locker.*befestigung)\b', 4),
                (r'\b(?:seil.*führung|führung.*seil)\b', 3),
                
                # Negative Indikatoren
                (r'\b(?:kabine|tür|fahrgast)\b', -2),
                (r'\b(?:motor|getriebe|hydraulik)\b', -1),
            ],
            
            "fahrstuhl_getriebe": [
                # Direkte Getriebe-Begriffe
                (r'\bgetriebe\b', 5),
                (r'\bmotor\b', 4),
                (r'\bantrieb\b', 4),
                (r'\bhydraulik\b', 4),
                (r'\bmaschinenraum\b', 4),
                
                # Kontextuelle Begriffe
                (r'\b(?:öl.*niedrig|niedrig.*öl)\b', 4),
                (r'\b(?:überhitzt|temperatur.*hoch)\b', 4),
                (r'\b(?:geräusch.*motor|motor.*geräusch)\b', 4),
                (r'\b(?:druck.*hydraulik|hydraulik.*druck)\b', 3),
                (r'\b(?:pumpe|ventilator|kühlung)\b', 3),
                
                # Negative Indikatoren
                (r'\b(?:kabine|fahrgast|notruf)\b', -2),
                (r'\b(?:seil|spannung|umlenkrolle)\b', -1),
            ]
        }
        
        logger.info("Advanced LLM-Klassifikator bereit")
    
    def classify(self, text: str) -> ClassificationResult:
        """Erweiterte regelbasierte Klassifikation"""
        text_lower = text.lower()
        
        scores = {"fahrstuhl_kabine": 0, "fahrstuhl_seil": 0, "fahrstuhl_getriebe": 0}
        
        for category, rules in self.classification_rules.items():
            for pattern, weight in rules:
                matches = len(re.findall(pattern, text_lower))
                scores[category] += matches * weight
        
        # Normalisierung und Konfidenz-Berechnung
        max_score = max(scores.values()) if scores.values() else 0
        
        if max_score <= 0:
            return ClassificationResult(
                category="kein_fokus",
                confidence=0.85,
                classifier_type="advanced_llm"
            )
        
        best_category = max(scores, key=scores.get)
        
        # Konfidenz basierend auf Score-Verteilung
        total_positive_score = sum(max(0, score) for score in scores.values())
        if total_positive_score > 0:
            confidence = min(0.95, (scores[best_category] / total_positive_score) * 0.8 + 0.2)
        else:
            confidence = 0.5
        
        return ClassificationResult(
            category=best_category,
            confidence=round(confidence, 3),
            classifier_type="advanced_llm"
        )

class EnhancedRuleBasedClassifier:
    """Verbesserte regelbasierte Klassifikation"""
    
    def __init__(self):
        logger.info("Initialisiere Enhanced Rule-Based Klassifikator...")
        
        # Mehrschichtige Regeln
        self.primary_rules = {
            "fahrstuhl_kabine": [
                r'\bkabine\b', r'\btür(?:en?)?\b', r'\bnotruf\b', 
                r'\bfahrgast\b', r'\bperson\b', r'\bpassagier\b'
            ],
            "fahrstuhl_seil": [
                r'\bseil(?:e)?\b', r'\bspannung\b', r'\bumlenkrolle\b',
                r'\baufhängung\b', r'\btragsystem\b', r'\btragseil\b'
            ],
            "fahrstuhl_getriebe": [
                r'\bgetriebe\b', r'\bmotor\b', r'\bantrieb\b',
                r'\bhydraulik\b', r'\bmaschinenraum\b', r'\bpumpe\b'
            ]
        }
        
        self.secondary_rules = {
            "fahrstuhl_kabine": [
                r'\bgewicht\b', r'\büberlast\b', r'\blicht\b', r'\binnenraum\b'
            ],
            "fahrstuhl_seil": [
                r'\bmaterialermüdung\b', r'\bbefestigung\b', r'\bführung\b'
            ],
            "fahrstuhl_getriebe": [
                r'\böl\b', r'\btemperatur\b', r'\bgeräusch\b', r'\bdruck\b'
            ]
        }
        
        logger.info("Enhanced Rule-Based Klassifikator bereit")
    
    def classify(self, text: str) -> ClassificationResult:
        """Klassifikation mit primären und sekundären Regeln"""
        text_lower = text.lower()
        
        # Primäre Matches (hohe Gewichtung)
        primary_scores = {}
        for category, patterns in self.primary_rules.items():
            primary_scores[category] = sum(
                len(re.findall(pattern, text_lower)) for pattern in patterns
            )
        
        # Sekundäre Matches (niedrige Gewichtung)  
        secondary_scores = {}
        for category, patterns in self.secondary_rules.items():
            secondary_scores[category] = sum(
                len(re.findall(pattern, text_lower)) for pattern in patterns
            ) * 0.5  # Reduzierte Gewichtung
        
        # Kombinierte Scores
        total_scores = {}
        for category in primary_scores:
            total_scores[category] = primary_scores[category] + secondary_scores[category]
        
        max_score = max(total_scores.values()) if total_scores.values() else 0
        
        if max_score == 0:
            return ClassificationResult(
                category="kein_fokus",
                confidence=0.9,
                classifier_type="enhanced_rules"
            )
        
        best_category = max(total_scores, key=total_scores.get)
        
        # Konfidenz basierend auf primären vs. sekundären Matches
        primary_contribution = primary_scores[best_category] / max(max_score, 1)
        confidence = min(0.95, 0.6 + (primary_contribution * 0.3) + (max_score * 0.05))
        
        return ClassificationResult(
            category=best_category,
            confidence=round(confidence, 3),
            classifier_type="enhanced_rules"
        )

class SimpleMultiClassifierSystem:
    """Vereinfachtes Multi-Klassifikator System ohne externe ML-Libraries"""
    
    def __init__(self):
        logger.info("Initialisiere Simple Multi-Klassifikator-System...")
        self.embedding_classifier = SimpleEmbeddingClassifier()
        self.llm_classifier = AdvancedLLMClassifier()
        self.rule_classifier = EnhancedRuleBasedClassifier()
        
        # Für BERT-Training
        self.training_data = []
        self.training_file = "bert_training_data.csv"
        
        logger.info("Simple Multi-Klassifikator-System bereit")
    
    def process_event(self, event: Event) -> ProcessedEvent:
        """Verarbeitet ein Event mit allen drei Klassifikatoren"""
        logger.info(f"Verarbeite Event: {event.id}")
        
        # Alle drei Klassifikatoren anwenden
        embedding_result = self.embedding_classifier.classify(event.raw_text)
        llm_result = self.llm_classifier.classify(event.raw_text)
        rule_result = self.rule_classifier.classify(event.raw_text)
        
        # Konsensus bestimmen
        consensus, certainty_level = self._determine_consensus(
            embedding_result, llm_result, rule_result
        )
        
        # Sollte für Training gespeichert werden?
        should_save_for_training = self._should_save_for_training(
            embedding_result, llm_result, rule_result, certainty_level
        )
        
        processed_event = ProcessedEvent(
            event=event,
            embedding_result=embedding_result,
            llm_result=llm_result,
            rule_result=rule_result,
            consensus=consensus,
            certainty_level=certainty_level,
            should_save_for_training=should_save_for_training
        )
        
        # Für BERT-Training speichern wenn alle übereinstimmen
        if should_save_for_training:
            self._save_for_bert_training(event.raw_text, consensus)
        
        self._log_results(processed_event)
        
        return processed_event
    
    def _determine_consensus(self, emb_result: ClassificationResult, 
                           llm_result: ClassificationResult, 
                           rule_result: ClassificationResult) -> Tuple[str, str]:
        """Bestimmt Konsensus und Sicherheitslevel"""
        
        categories = [emb_result.category, llm_result.category, rule_result.category]
        unique_categories = set(categories)
        
        # Alle drei stimmen überein
        if len(unique_categories) == 1:
            return categories[0], "certain"
        
        # Zwei stimmen überein
        elif len(unique_categories) == 2:
            category_counts = Counter(categories)
            consensus_category = category_counts.most_common(1)[0][0]
            return consensus_category, "uncertain"
        
        # Alle drei unterschiedlich
        else:
            # Weighted Voting basierend auf Konfidenz
            weighted_scores = {}
            for result in [emb_result, llm_result, rule_result]:
                if result.category not in weighted_scores:
                    weighted_scores[result.category] = 0
                weighted_scores[result.category] += result.confidence
            
            best_category = max(weighted_scores, key=weighted_scores.get)
            return best_category, "very_uncertain"
    
    def _should_save_for_training(self, emb_result: ClassificationResult,
                                llm_result: ClassificationResult,
                                rule_result: ClassificationResult,
                                certainty_level: str) -> bool:
        """Bestimmt ob das Event für BERT-Training gespeichert werden soll"""
        return (certainty_level == "certain" and 
                emb_result.category == llm_result.category == rule_result.category)
    
    def _save_for_bert_training(self, text: str, label: str):
        """Speichert Text und Label für BERT-Training"""
        self.training_data.append({"text": text, "label": label})
        
        # Schreibe in CSV-Datei
        file_exists = os.path.exists(self.training_file)
        with open(self.training_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["text", "label"])
            writer.writerow([text, label])
        
        logger.info(f"Für BERT-Training gespeichert: {label}")
    
    def _log_results(self, processed_event: ProcessedEvent):
        """Loggt die Ergebnisse"""
        event = processed_event.event
        logger.info(f"Event {event.id} - Konsensus: {processed_event.consensus} "
                   f"({processed_event.certainty_level})")
        logger.info(f"  Embedding: {processed_event.embedding_result.category} "
                   f"({processed_event.embedding_result.confidence})")
        logger.info(f"  LLM: {processed_event.llm_result.category} "
                   f"({processed_event.llm_result.confidence})")
        logger.info(f"  Rules: {processed_event.rule_result.category} "
                   f"({processed_event.rule_result.confidence})")
        
        if processed_event.should_save_for_training:
            logger.info("  ✓ Für BERT-Training gespeichert")
    
    def get_training_data_stats(self) -> Dict:
        """Gibt Statistiken über die gesammelten Trainingsdaten zurück"""
        if not os.path.exists(self.training_file):
            return {"total": 0, "by_category": {}}
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        label_counts = Counter(row['label'] for row in data)
        
        return {
            "total": len(data),
            "by_category": dict(label_counts)
        }

# Backwards compatibility - verwende Simple System als Standard
MultiClassifierSystem = SimpleMultiClassifierSystem

# Test des Systems
if __name__ == "__main__":
    print(" Teste Simple Multi-Classifier System...")
    
    system = SimpleMultiClassifierSystem()
    
    test_events = [
        Event("1", 1234567890, "sensor", "warnung", 
              "Kabinentür schließt verzögert nach Gewichtserkennung"),
        Event("2", 1234567891, "funk", "bedrohung", 
              "Seilspannung kritisch, mögliche Überlastung"),
        Event("3", 1234567892, "email", "info", 
              "Getriebeölstand nahe Mindestwert"),
    ]
    
    for event in test_events:
        result = system.process_event(event)
        print(f" Event {event.id}: {result.consensus} ({result.certainty_level})")
    
    print(" Simple System funktioniert!")