# enhanced_multi_classifier.py - Erweitertes System mit Fokus-Modi und Performance-Tracking
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

from event import Event, ClassificationResult
from base_classifier import BaseClassifier
from simple_classifier import SimpleEmbeddingClassifier
from rule_classifier import EnhancedRuleBasedClassifier
from ollama_classifier import OllamaLangChainClassifier
from tfidf_classifier import TfidfMLClassifier
from zero_shot_classifier import ZeroShotClassifier

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Bedrohungsstufen"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class FocusMode(Enum):
    """System-Fokus-Modi"""
    NORMAL = "normal"
    FAHRKABINE_FOCUS = "fahrkabine_focus"
    SEIL_FOCUS = "seil_focus"
    GETRIEBE_FOCUS = "getriebe_focus"
    EMERGENCY_ALL = "emergency_all"

@dataclass
class ThreatIndicator:
    """Bedrohungsindikator"""
    keywords: List[str]
    level: ThreatLevel
    weight: float

@dataclass
class ClassifierMetrics:
    """Performance-Metriken für einen Klassifikator"""
    name: str
    true_positives: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    false_positives: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    false_negatives: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_predictions: int = 0
    correct_predictions: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    
    def update(self, predicted: str, actual: str, confidence: float, processing_time: float):
        """Aktualisiert Metriken mit neuer Vorhersage"""
        self.total_predictions += 1
        self.confidence_scores.append(confidence)
        self.processing_times.append(processing_time)
        
        if predicted == actual:
            self.correct_predictions += 1
            self.true_positives[actual] += 1
        else:
            self.false_positives[predicted] += 1
            self.false_negatives[actual] += 1
    
    def get_accuracy(self) -> float:
        """Berechnet Gesamtgenauigkeit"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def get_precision(self, category: str) -> float:
        """Berechnet Precision für eine Kategorie"""
        tp = self.true_positives[category]
        fp = self.false_positives[category]
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    def get_recall(self, category: str) -> float:
        """Berechnet Recall für eine Kategorie"""
        tp = self.true_positives[category]
        fn = self.false_negatives[category]
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    def get_f1_score(self, category: str) -> float:
        """Berechnet F1-Score für eine Kategorie"""
        precision = self.get_precision(category)
        recall = self.get_recall(category)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_avg_confidence(self) -> float:
        """Durchschnittliche Konfidenz"""
        if not self.confidence_scores:
            return 0.0
        return np.mean(self.confidence_scores)
    
    def get_avg_processing_time(self) -> float:
        """Durchschnittliche Verarbeitungszeit"""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times)

class EnhancedMultiClassifierSystem:
    """Erweitertes Multi-Klassifikator System mit Fokus-Modi und Performance-Tracking"""
    
    def __init__(self):
        self.classifiers: List[BaseClassifier] = []
        self.current_mode = FocusMode.NORMAL
        self.threat_level = ThreatLevel.NORMAL
        
        # Historie für Warnung-Tracking
        self.category_warnings = defaultdict(list)
        self.warning_window = timedelta(minutes=30)
        self.warning_threshold = 3
        
        # Bedrohungsindikatoren
        self.threat_indicators = [
            ThreatIndicator(
                keywords=['notfall', 'emergency', 'kritisch', 'sofort', 'gefahr', 'ausfall'],
                level=ThreatLevel.EMERGENCY,
                weight=1.0
            ),
            ThreatIndicator(
                keywords=['plötzlich', 'unerwartet', 'stillstand', 'blockiert', 'defekt'],
                level=ThreatLevel.CRITICAL,
                weight=0.8
            ),
            ThreatIndicator(
                keywords=['verzögert', 'langsam', 'unregelmäßig', 'vibriert', 'geräusch'],
                level=ThreatLevel.WARNING,
                weight=0.5
            )
        ]
        
        # Performance Tracking
        self.classification_history = []
        self.mode_changes = []
        
        # Statistik-Tracking
        self.total_events = 0
        self.total_time = 0.0
        
        # Classifier Performance Metrics
        self.classifier_metrics: Dict[str, ClassifierMetrics] = {}
        
        # Ground Truth Storage (für supervised learning)
        self.ground_truth_labels: Dict[str, str] = {}
        
        # Kategorien für Tracking
        self.categories = ['fahrkabine', 'seil', 'aufzugsgetriebe', 'unknown']
    
    async def initialize(self):
        """Initialisiert alle Klassifikatoren"""
        logger.info("Initialisiere Async Multi-Klassifikator-System...")
        
        # Erstelle Klassifikatoren
        self.classifiers = [
            SimpleEmbeddingClassifier(),
            EnhancedRuleBasedClassifier(),
            TfidfMLClassifier(training_data={
                "Seil quietscht beim Aufwärtsfahren": "seil",
                "Kabinentür klemmt beim Schließen": "fahrkabine",
                "Motor überhitzt nach kurzer Laufzeit": "aufzugsgetriebe"
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
        
        # Prüfe Ergebnisse und initialisiere Metriken
        active_classifiers = []
        for i, (classifier, result) in enumerate(zip(self.classifiers, results)):
            if isinstance(result, Exception):
                logger.warning(f"Klassifikator {classifier.name} Initialisierung fehlgeschlagen: {result}")
            elif result:
                active_classifiers.append(classifier)
                # Initialisiere Metriken für diesen Klassifikator
                self.classifier_metrics[classifier.name] = ClassifierMetrics(name=classifier.name)
                logger.info(f"✅ {classifier.name} erfolgreich initialisiert")
            else:
                logger.warning(f"Klassifikator {classifier.name} nicht verfügbar")
        
        self.classifiers = active_classifiers
        logger.info("Async Multi-Klassifikator-System bereit")
    
    def set_ground_truth(self, message: str, true_category: str):
        """Setzt die wahre Kategorie für eine Nachricht (für Evaluation)"""
        self.ground_truth_labels[message.lower().strip()] = true_category
    
    def load_ground_truth(self, ground_truth_data: Dict[str, str]):
        """Lädt Ground Truth Daten für Evaluation"""
        for message, category in ground_truth_data.items():
            self.set_ground_truth(message, category)
    
    async def classify_event(self, event: Event, true_category: Optional[str] = None) -> Dict[str, Any]:
        """Klassifiziert Event mit Fokus-Modus-Logik und Performance-Tracking"""
        start_time = asyncio.get_event_loop().time()
        
        # Versuche Ground Truth zu finden wenn nicht angegeben
        if true_category is None:
            message_key = event.message.lower().strip()
            true_category = self.ground_truth_labels.get(message_key)
        
        # 1. Bedrohungsanalyse
        threat_analysis = self._analyze_threat(event.message)
        
        # 2. Führe Klassifikation durch
        classification_tasks = []
        for classifier in self.classifiers:
            task = asyncio.create_task(classifier.classify_with_timing(event))
            classification_tasks.append(task)
        
        results = await asyncio.gather(*classification_tasks, return_exceptions=True)
        
        # 3. Verarbeite Ergebnisse und update Metriken
        valid_results = []
        individual_scores = {}
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                valid_results.append(result)
                classifier = self.classifiers[i]
                
                # Speichere individuelle Klassifikator-Ergebnisse
                individual_scores[classifier.name] = {
                    'categories': result.categories,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                }
                
                # Update Metriken wenn Ground Truth verfügbar
                if true_category and true_category != 'unknown':
                    predicted_category = max(result.categories, key=result.categories.get)
                    metrics = self.classifier_metrics[classifier.name]
                    metrics.update(
                        predicted=predicted_category,
                        actual=true_category,
                        confidence=result.confidence,
                        processing_time=result.processing_time
                    )
        
        # 4. Kombiniere Ergebnisse
        combined_scores = self._combine_results(valid_results)
        
        # 5. Update Modus basierend auf Ergebnissen
        await self._update_focus_mode(combined_scores, threat_analysis)
        
        # 6. Erstelle detaillierte Antwort
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Update Statistiken
        self.total_events += 1
        self.total_time += processing_time
        
        # Finale Vorhersage
        final_prediction = max(combined_scores, key=combined_scores.get)
        
        response = {
            'event': event.to_dict(),
            'combined_score': combined_scores,
            'individual_scores': individual_scores,
            'threat_analysis': {
                'level': threat_analysis['level'].value,
                'indicators': threat_analysis['indicators'],
                'confidence': threat_analysis['confidence']
            },
            'system_status': {
                'mode': self.current_mode.value,
                'threat_level': self.threat_level.value,
                'recent_warnings': self._get_recent_warnings()
            },
            'processing_time': processing_time,
            'classifier_count': len(valid_results),
            'final_prediction': final_prediction,
            'true_category': true_category  # None wenn nicht verfügbar
        }
        
        # 7. Speichere in Historie
        self.classification_history.append({
            'timestamp': datetime.now(),
            'category': final_prediction,
            'true_category': true_category,
            'confidence': max(combined_scores.values()),
            'threat_level': threat_analysis['level']
        })
        
        return response
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gibt detaillierte Performance-Metriken zurück"""
        metrics = {}
        
        # Gesamtsystem-Metriken
        total_correct = sum(1 for h in self.classification_history 
                           if h.get('true_category') and h['category'] == h['true_category'])
        total_with_truth = sum(1 for h in self.classification_history if h.get('true_category'))
        
        system_accuracy = total_correct / total_with_truth if total_with_truth > 0 else 0.0
        
        metrics['system'] = {
            'total_classifications': len(self.classification_history),
            'classifications_with_ground_truth': total_with_truth,
            'overall_accuracy': system_accuracy,
            'average_processing_time': self.total_time / max(1, self.total_events)
        }
        
        # Einzelne Klassifikator-Metriken
        metrics['classifiers'] = {}
        
        for name, clf_metrics in self.classifier_metrics.items():
            classifier_data = {
                'accuracy': clf_metrics.get_accuracy(),
                'total_predictions': clf_metrics.total_predictions,
                'correct_predictions': clf_metrics.correct_predictions,
                'avg_confidence': clf_metrics.get_avg_confidence(),
                'avg_processing_time': clf_metrics.get_avg_processing_time(),
                'categories': {}
            }
            
            # Metriken pro Kategorie
            for category in self.categories:
                if category != 'unknown':
                    classifier_data['categories'][category] = {
                        'precision': clf_metrics.get_precision(category),
                        'recall': clf_metrics.get_recall(category),
                        'f1_score': clf_metrics.get_f1_score(category),
                        'true_positives': clf_metrics.true_positives[category],
                        'false_positives': clf_metrics.false_positives[category],
                        'false_negatives': clf_metrics.false_negatives[category]
                    }
            
            metrics['classifiers'][name] = classifier_data
        
        # Konfusionsmatrix für Gesamtsystem
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for entry in self.classification_history:
            if entry.get('true_category'):
                predicted = entry['category']
                actual = entry['true_category']
                confusion_matrix[actual][predicted] += 1
        
        metrics['confusion_matrix'] = dict(confusion_matrix)
        
        return metrics
    
    def print_performance_report(self):
        """Druckt einen detaillierten Performance-Report"""
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*80)
        print("PERFORMANCE REPORT")
        print("="*80)
        
        # System-Übersicht
        sys_metrics = metrics['system']
        print(f"\nSYSTEM-ÜBERSICHT:")
        print(f"  Gesamte Klassifikationen: {sys_metrics['total_classifications']}")
        print(f"  Mit Ground Truth: {sys_metrics['classifications_with_ground_truth']}")
        print(f"  Gesamt-Genauigkeit: {sys_metrics['overall_accuracy']*100:.1f}%")
        print(f"  Durchschn. Verarbeitungszeit: {sys_metrics['average_processing_time']*1000:.1f}ms")
        
        # Klassifikator-Details
        print("\nKLASSIFIKATOR-PERFORMANCE:")
        for clf_name, clf_data in metrics['classifiers'].items():
            print(f"\n  {clf_name}:")
            print(f"    Genauigkeit: {clf_data['accuracy']*100:.1f}%")
            print(f"    Durchschn. Konfidenz: {clf_data['avg_confidence']*100:.1f}%")
            print(f"    Durchschn. Zeit: {clf_data['avg_processing_time']*1000:.1f}ms")
            
            # Kategorie-Details
            print("    Kategorie-Metriken:")
            for cat, cat_metrics in clf_data['categories'].items():
                f1 = cat_metrics['f1_score']
                if f1 > 0:  # Nur Kategorien mit Aktivität
                    print(f"      {cat}: F1={f1:.2f}, Präz={cat_metrics['precision']:.2f}, Recall={cat_metrics['recall']:.2f}")
        
        # Konfusionsmatrix
        if metrics['confusion_matrix']:
            print("\nKONFUSIONSMATRIX (Actual vs Predicted):")
            categories = list(set(list(metrics['confusion_matrix'].keys()) + 
                                 [p for preds in metrics['confusion_matrix'].values() for p in preds.keys()]))
            
            # Header
            print(f"{'':15s}", end='')
            for cat in categories:
                print(f"{cat[:10]:>12s}", end='')
            print()
            
            # Zeilen
            for actual in categories:
                print(f"{actual:15s}", end='')
                for predicted in categories:
                    count = metrics['confusion_matrix'].get(actual, {}).get(predicted, 0)
                    print(f"{count:12d}", end='')
                print()
        
        print("\n" + "="*80)
    
    def _analyze_threat(self, message: str) -> Dict[str, Any]:
        """Analysiert Bedrohungslevel der Nachricht"""
        message_lower = message.lower()
        detected_indicators = []
        max_level = ThreatLevel.NORMAL
        total_weight = 0.0
        
        for indicator in self.threat_indicators:
            for keyword in indicator.keywords:
                if keyword in message_lower:
                    detected_indicators.append(keyword)
                    total_weight += indicator.weight
                    if indicator.level.value > max_level.value:
                        max_level = indicator.level
        
        confidence = min(total_weight, 1.0)
        
        return {
            'level': max_level,
            'indicators': detected_indicators,
            'confidence': confidence
        }
    
    async def _update_focus_mode(self, scores: Dict[str, float], threat_analysis: Dict[str, Any]):
        """Aktualisiert System-Fokus basierend auf Ergebnissen"""
        old_mode = self.current_mode
        
        # 1. Prüfe auf Notfall
        if threat_analysis['level'] == ThreatLevel.EMERGENCY:
            self.current_mode = FocusMode.EMERGENCY_ALL
            self.threat_level = ThreatLevel.EMERGENCY
            logger.warning(f"🚨 NOTFALL ERKANNT! Wechsle zu {self.current_mode.value}")
        
        # 2. Prüfe auf kritische Bedrohung
        elif threat_analysis['level'] == ThreatLevel.CRITICAL:
            best_category = max(scores, key=scores.get)
            if best_category == 'fahrkabine':
                self.current_mode = FocusMode.FAHRKABINE_FOCUS
            elif best_category == 'seil':
                self.current_mode = FocusMode.SEIL_FOCUS
            elif best_category == 'aufzugsgetriebe':
                self.current_mode = FocusMode.GETRIEBE_FOCUS
            
            self.threat_level = ThreatLevel.CRITICAL
            logger.warning(f"⚠️ Kritische Situation! Fokus auf {best_category}")
        
        # 3. Prüfe auf wiederholte Warnungen
        else:
            if threat_analysis['level'] == ThreatLevel.WARNING:
                best_category = max(scores, key=scores.get)
                if best_category != 'unknown':
                    self.category_warnings[best_category].append(datetime.now())
            
            for category, warnings in self.category_warnings.items():
                recent_warnings = self._count_recent_warnings(warnings)
                if recent_warnings >= self.warning_threshold:
                    if category == 'fahrkabine':
                        self.current_mode = FocusMode.FAHRKABINE_FOCUS
                    elif category == 'seil':
                        self.current_mode = FocusMode.SEIL_FOCUS
                    elif category == 'aufzugsgetriebe':
                        self.current_mode = FocusMode.GETRIEBE_FOCUS
                    
                    self.threat_level = ThreatLevel.WARNING
                    logger.info(f"📊 Mehrere Warnungen für {category} - Wechsle zu Fokus-Modus")
                    break
            else:
                if self.threat_level != ThreatLevel.NORMAL:
                    if self._can_deescalate():
                        self.current_mode = FocusMode.NORMAL
                        self.threat_level = ThreatLevel.NORMAL
                        logger.info("✅ Situation normalisiert - Zurück zu Normal-Modus")
        
        if old_mode != self.current_mode:
            self.mode_changes.append({
                'timestamp': datetime.now(),
                'from': old_mode.value,
                'to': self.current_mode.value,
                'reason': f"Threat: {threat_analysis['level'].value}"
            })
    
    def _count_recent_warnings(self, warnings: List[datetime]) -> int:
        """Zählt Warnungen im Zeitfenster"""
        cutoff = datetime.now() - self.warning_window
        return sum(1 for w in warnings if w > cutoff)
    
    def _get_recent_warnings(self) -> Dict[str, int]:
        """Gibt aktuelle Warnungszähler zurück"""
        result = {}
        for category, warnings in self.category_warnings.items():
            result[category] = self._count_recent_warnings(warnings)
        return result
    
    def _can_deescalate(self) -> bool:
        """Prüft ob System de-eskalieren kann"""
        cutoff = datetime.now() - timedelta(minutes=15)
        
        for warnings in self.category_warnings.values():
            if any(w > cutoff for w in warnings):
                return False
        
        recent_history = [h for h in self.classification_history[-10:] 
                         if h['timestamp'] > cutoff]
        
        return not any(h['threat_level'] in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY] 
                      for h in recent_history)
    
    def _combine_results(self, results: List[ClassificationResult]) -> Dict[str, float]:
        """Kombiniert Ergebnisse mit Modus-Gewichtung"""
        if not results:
            return {'unknown': 1.0}
        
        all_categories = set()
        for result in results:
            all_categories.update(result.categories.keys())
        
        combined = {}
        for category in all_categories:
            scores = []
            weights = []
            
            for result in results:
                if category in result.categories:
                    score = result.categories[category]
                    weight = result.confidence
                    
                    # Modus-basierte Gewichtung
                    if self.current_mode == FocusMode.FAHRKABINE_FOCUS and category == 'fahrkabine':
                        weight *= 1.5
                    elif self.current_mode == FocusMode.SEIL_FOCUS and category == 'seil':
                        weight *= 1.5
                    elif self.current_mode == FocusMode.GETRIEBE_FOCUS and category == 'aufzugsgetriebe':
                        weight *= 1.5
                    elif self.current_mode == FocusMode.EMERGENCY_ALL:
                        weight *= 2.0
                    
                    scores.append(score)
                    weights.append(weight)
                else:
                    scores.append(0.0)
                    weights.append(0.1)
            
            if sum(weights) > 0:
                combined[category] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                combined[category] = sum(scores) / len(scores)
        
        # Normalisiere auf Summe 1.0
        total = sum(combined.values())
        if total > 0:
            for cat in combined:
                combined[cat] /= total
        
        return combined
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Erweiterte System-Statistiken"""
        base_stats = self.get_system_stats()
        
        # Füge erweiterte Statistiken hinzu
        base_stats['enhanced_info'] = {
            'current_mode': self.current_mode.value,
            'threat_level': self.threat_level.value,
            'mode_changes': len(self.mode_changes),
            'recent_mode_changes': self.mode_changes[-5:],
            'warnings_by_category': self._get_recent_warnings(),
            'classification_history_size': len(self.classification_history)
        }
        
        # Füge Performance-Metriken hinzu wenn verfügbar
        if any(m.total_predictions > 0 for m in self.classifier_metrics.values()):
            base_stats['performance_summary'] = {
                clf_name: {
                    'accuracy': metrics.get_accuracy(),
                    'predictions': metrics.total_predictions
                }
                for clf_name, metrics in self.classifier_metrics.items()
            }
        
        return base_stats
    
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