# enhanced_multi_classifier.py - Erweitertes System mit Fokus-Modi
import asyncio
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta

from event import Event, ClassificationResult
from base_classifier import BaseClassifier

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

class EnhancedMultiClassifierSystem:
    """Erweitertes Multi-Klassifikator System mit Fokus-Modi"""
    
    def __init__(self):
        self.classifiers: List[BaseClassifier] = []
        self.current_mode = FocusMode.NORMAL
        self.threat_level = ThreatLevel.NORMAL
        
        # Historie für Warnung-Tracking
        self.category_warnings = defaultdict(list)
        self.warning_window = timedelta(minutes=30)
        self.warning_threshold = 3  # Anzahl Warnungen für Fokus-Wechsel
        
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
    
    async def classify_event(self, event: Event) -> Dict[str, Any]:
        """Klassifiziert Event mit Fokus-Modus-Logik"""
        start_time = asyncio.get_event_loop().time()
        
        # 1. Bedrohungsanalyse
        threat_analysis = self._analyze_threat(event.message)
        
        # 2. Führe Klassifikation durch
        classification_tasks = []
        for classifier in self.classifiers:
            task = asyncio.create_task(classifier.classify_with_timing(event))
            classification_tasks.append(task)
        
        results = await asyncio.gather(*classification_tasks, return_exceptions=True)
        
        # 3. Verarbeite Ergebnisse
        valid_results = []
        individual_scores = {}
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                valid_results.append(result)
                # Speichere individuelle Klassifikator-Ergebnisse
                individual_scores[self.classifiers[i].name] = {
                    'categories': result.categories,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                }
        
        # 4. Kombiniere Ergebnisse
        combined_scores = self._combine_results(valid_results)
        
        # 5. Update Modus basierend auf Ergebnissen
        await self._update_focus_mode(combined_scores, threat_analysis)
        
        # 6. Erstelle detaillierte Antwort
        processing_time = asyncio.get_event_loop().time() - start_time
        
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
            'classifier_count': len(valid_results)
        }
        
        # 7. Speichere in Historie
        self.classification_history.append({
            'timestamp': datetime.now(),
            'category': max(combined_scores, key=combined_scores.get),
            'confidence': max(combined_scores.values()),
            'threat_level': threat_analysis['level']
        })
        
        return response
    
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
            # Fokussiere auf betroffene Komponente
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
            # Update Warnungs-Historie
            if threat_analysis['level'] == ThreatLevel.WARNING:
                best_category = max(scores, key=scores.get)
                if best_category != 'unknown':
                    self.category_warnings[best_category].append(datetime.now())
            
            # Prüfe ob Schwellwert überschritten
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
                # Keine besonderen Bedingungen - Normal-Modus
                if self.threat_level != ThreatLevel.NORMAL:
                    # Prüfe ob wir de-eskalieren können
                    if self._can_deescalate():
                        self.current_mode = FocusMode.NORMAL
                        self.threat_level = ThreatLevel.NORMAL
                        logger.info("✅ Situation normalisiert - Zurück zu Normal-Modus")
        
        # Logge Modus-Änderung
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
        # Keine Warnungen in den letzten 15 Minuten
        cutoff = datetime.now() - timedelta(minutes=15)
        
        for warnings in self.category_warnings.values():
            if any(w > cutoff for w in warnings):
                return False
        
        # Keine kritischen Events in Historie
        recent_history = [h for h in self.classification_history[-10:] 
                         if h['timestamp'] > cutoff]
        
        return not any(h['threat_level'] in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY] 
                      for h in recent_history)
    
    def _combine_results(self, results: List[ClassificationResult]) -> Dict[str, float]:
        """Kombiniert Ergebnisse mit Modus-Gewichtung"""
        if not results:
            return {'unknown': 1.0}
        
        # Basis-Kombination
        combined = {}
        all_categories = set()
        for result in results:
            all_categories.update(result.categories.keys())
        
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
                        weight *= 2.0  # Erhöhe alle Gewichte im Notfall
                    
                    scores.append(score)
                    weights.append(weight)
                else:
                    scores.append(0.0)
                    weights.append(0.1)
            
            if sum(weights) > 0:
                combined[category] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            else:
                combined[category] = sum(scores) / len(scores)
        
        # Normalisiere auf max 1.0
        max_score = max(combined.values()) if combined else 1.0
        if max_score > 1.0:
            for cat in combined:
                combined[cat] /= max_score
        
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
        
        return base_stats