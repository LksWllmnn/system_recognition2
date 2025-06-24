# enhanced_logging.py - Verbessertes Logging und Visualisierung mit Performance-Tracking
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from colorama import Fore, Back, Style, init
import json
from collections import defaultdict

# Initialisiere Colorama für Windows-Kompatibilität
init(autoreset=True)

class ClassificationLogger:
    """Spezieller Logger für Klassifikationsergebnisse mit Performance-Tracking"""
    
    def __init__(self, name: str = "ClassificationLogger"):
        self.logger = logging.getLogger(name)
        self.results_history = []
        self.performance_tracking_enabled = False
        self.classifier_performance = defaultdict(lambda: {
            'total': 0,
            'correct': 0,
            'predictions': defaultdict(lambda: defaultdict(int))
        })
    
    def enable_performance_tracking(self):
        """Aktiviert Performance-Tracking"""
        self.performance_tracking_enabled = True
        self.logger.info("Performance-Tracking aktiviert")
    
    def log_classification_result(self, result: Dict[str, Any]):
        """Loggt detaillierte Klassifikationsergebnisse"""
        # Speichere in Historie
        self.results_history.append({
            'timestamp': datetime.now(),
            'result': result
        })
        
        # Extrahiere Informationen
        event = result.get('event', {})
        message = event.get('message', 'N/A')
        combined_scores = result.get('combined_score', {})
        individual_scores = result.get('individual_scores', {})
        threat_info = result.get('threat_analysis', {})
        system_status = result.get('system_status', {})
        
        # Performance-Tracking wenn verfügbar
        true_category = result.get('true_category')
        final_prediction = result.get('final_prediction')
        
        # Finde beste Kategorie
        best_category = max(combined_scores, key=combined_scores.get) if combined_scores else 'unknown'
        best_score = combined_scores.get(best_category, 0)
        
        # Formatiere Ausgabe
        print("\n" + "="*80)
        print(f"{Fore.CYAN}KLASSIFIKATIONSERGEBNIS")
        print("="*80)
        
        # Nachricht
        print(f"\nNachricht: {Style.BRIGHT}{message}{Style.RESET_ALL}")
        
        # Kanal-Info wenn vorhanden
        metadata = event.get('metadata', {})
        if 'channel' in metadata:
            priority_colors = {0: Fore.GREEN, 1: Fore.YELLOW, 2: Fore.RED}
            priority = metadata.get('priority', 0)
            color = priority_colors.get(priority, Fore.WHITE)
            print(f"Kanal: {metadata['channel']} | {color}Priorität: {priority}{Fore.RESET}")
        
        # Finale Klassifikation
        category_colors = {
            'fahrkabine': Fore.BLUE,
            'seil': Fore.MAGENTA,
            'aufzugsgetriebe': Fore.CYAN,
            'unknown': Fore.LIGHTBLACK_EX
        }
        
        color = category_colors.get(best_category, Fore.WHITE)
        print(f"\n{Fore.GREEN}FINALE KLASSIFIKATION: {color}{Style.BRIGHT}{best_category.upper()}{Style.RESET_ALL} ({best_score*100:.1f}%)")
        
        # Ground Truth wenn verfügbar
        if true_category:
            truth_color = category_colors.get(true_category, Fore.WHITE)
            if best_category == true_category:
                print(f"{Fore.GREEN}GROUND TRUTH: {truth_color}{true_category.upper()}{Fore.RESET} - KORREKT!")
            else:
                print(f"{Fore.RED}GROUND TRUTH: {truth_color}{true_category.upper()}{Fore.RESET} - FALSCH!")
            
            # Update Performance-Tracking
            if self.performance_tracking_enabled:
                self._update_performance_metrics(individual_scores, true_category)
        
        # Bedrohungsanalyse
        if threat_info:
            threat_level = threat_info.get('level', 'normal')
            threat_colors = {
                'normal': Fore.GREEN,
                'warning': Fore.YELLOW,
                'critical': Fore.RED,
                'emergency': Back.RED + Fore.WHITE
            }
            threat_color = threat_colors.get(threat_level, Fore.WHITE)
            indicators = threat_info.get('indicators', [])
            
            print(f"\n{Fore.YELLOW}BEDROHUNGSANALYSE:")
            print(f"   Level: {threat_color}{threat_level.upper()}{Style.RESET_ALL}")
            if indicators:
                print(f"   Indikatoren: {', '.join(indicators)}")
            print(f"   Konfidenz: {threat_info.get('confidence', 0)*100:.1f}%")
        
        # System-Status
        if system_status:
            mode = system_status.get('mode', 'normal')
            mode_colors = {
                'normal': Fore.GREEN,
                'fahrkabine_focus': Fore.BLUE,
                'seil_focus': Fore.MAGENTA,
                'getriebe_focus': Fore.CYAN,
                'emergency_all': Back.RED + Fore.WHITE
            }
            mode_color = mode_colors.get(mode, Fore.WHITE)
            
            print(f"\n{Fore.CYAN}SYSTEM-STATUS:")
            print(f"   Modus: {mode_color}{mode.upper()}{Style.RESET_ALL}")
            
            warnings = system_status.get('recent_warnings', {})
            if warnings:
                print(f"   Aktuelle Warnungen:")
                for category, count in warnings.items():
                    if count > 0:
                        print(f"      - {category}: {count} Warnungen")
        
        # Einzelne Klassifikator-Ergebnisse (kompakt)
        print(f"\n{Fore.YELLOW}EINZELNE KLASSIFIKATOR-ERGEBNISSE:")
        
        for classifier_name, scores_data in individual_scores.items():
            categories = scores_data.get('categories', {})
            confidence = scores_data.get('confidence', 0)
            proc_time = scores_data.get('processing_time', 0)
            
            if categories:
                best_cat = max(categories, key=categories.get)
                best_score_clf = categories[best_cat]
                
                # Formatiere Klassifikator-Name
                classifier_display = {
                    'SimpleEmbedding': 'Einfach',
                    'EnhancedRuleBased': 'Regel-basiert',
                    'TfidfML': 'TfidfML',
                    'ZeroShot': 'ZeroShot',
                    'OllamaLLM': 'LLM (Ollama)'
                }.get(classifier_name, classifier_name)
                
                # Kompakte Darstellung mit Korrektheit-Indikator
                correctness = ""
                if true_category:
                    if best_cat == true_category:
                        correctness = f" {Fore.GREEN}✓{Fore.RESET}"
                    else:
                        correctness = f" {Fore.RED}✗{Fore.RESET}"
                
                print(f"\n   {classifier_display}:{correctness}")
                print(f"      Wahl: {best_cat} ({best_score_clf*100:.1f}%)")
                print(f"      Konfidenz: {confidence*100:.1f}%")
                print(f"      Zeit: {proc_time*1000:.1f}ms")
                
                # Zeige alle Scores
                sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
                scores_str = " | ".join([f"{cat}:{score*100:.0f}%" for cat, score in sorted_cats])
                print(f"      Scores: {scores_str}")
        
        # Kombinierte Scores mit Normalisierung
        print(f"\n{Fore.GREEN}KOMBINIERTE SCORES:")
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prüfe ob normalisiert
        total = sum(score for _, score in sorted_combined)
        if abs(total - 1.0) > 0.01:
            print(f"   {Fore.YELLOW}Warnung: Summe = {total*100:.1f}% (sollte 100% sein){Fore.RESET}")
        
        for cat, score in sorted_combined:
            if score > 0.01:  # Nur Scores über 1%
                bar_length = int(score * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                color = category_colors.get(cat, Fore.WHITE)
                print(f"   {color}{cat:15s}{Fore.RESET} [{bar}] {score*100:5.1f}%")
        
        # Performance
        total_time = result.get('processing_time', 0)
        print(f"\n{Fore.CYAN}PERFORMANCE:")
        print(f"   Gesamtzeit: {total_time*1000:.1f}ms")
        print(f"   Klassifikatoren: {result.get('classifier_count', 0)}")
        
        print("\n" + "="*80)
    
    def _update_performance_metrics(self, individual_scores: Dict[str, Any], true_category: str):
        """Aktualisiert Performance-Metriken für Klassifikatoren"""
        for classifier_name, scores_data in individual_scores.items():
            categories = scores_data.get('categories', {})
            if categories:
                predicted = max(categories, key=categories.get)
                
                # Update Zähler
                self.classifier_performance[classifier_name]['total'] += 1
                if predicted == true_category:
                    self.classifier_performance[classifier_name]['correct'] += 1
                
                # Konfusionsmatrix
                self.classifier_performance[classifier_name]['predictions'][true_category][predicted] += 1
    
    def log_mode_change(self, old_mode: str, new_mode: str, reason: str):
        """Loggt Modus-Änderungen"""
        print(f"\n{Back.YELLOW}{Fore.BLACK} MODUS-ÄNDERUNG {Style.RESET_ALL}")
        print(f"{old_mode} -> {new_mode}")
        print(f"Grund: {reason}\n")
    
    def generate_performance_report(self) -> str:
        """Generiert detaillierten Performance-Report"""
        if not self.performance_tracking_enabled:
            return "Performance-Tracking nicht aktiviert. Nutze enable_performance_tracking()"
        
        report = []
        report.append("\n" + "="*80)
        report.append("PERFORMANCE-ANALYSE")
        report.append("="*80)
        
        # Gesamtstatistiken
        total_with_truth = sum(1 for entry in self.results_history 
                              if entry['result'].get('true_category'))
        report.append(f"\nKlassifikationen mit Ground Truth: {total_with_truth}")
        
        if total_with_truth == 0:
            report.append("Keine Klassifikationen mit Ground Truth verfügbar.")
            return "\n".join(report)
        
        # Klassifikator-Performance
        report.append("\nKLASSIFIKATOR-GENAUIGKEIT:")
        
        classifiers_sorted = sorted(self.classifier_performance.items(), 
                                   key=lambda x: x[1]['correct']/max(1, x[1]['total']), 
                                   reverse=True)
        
        for clf_name, perf_data in classifiers_sorted:
            total = perf_data['total']
            correct = perf_data['correct']
            accuracy = correct / total if total > 0 else 0
            
            # Farbcodierung basierend auf Genauigkeit
            if accuracy >= 0.8:
                color = Fore.GREEN
            elif accuracy >= 0.6:
                color = Fore.YELLOW
            else:
                color = Fore.RED
            
            bar_length = int(accuracy * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            report.append(f"\n   {clf_name:20s} [{bar}] {color}{accuracy*100:5.1f}%{Fore.RESET}")
            report.append(f"   {'':20s} Korrekt: {correct}/{total}")
            
            # Mini-Konfusionsmatrix
            predictions = perf_data['predictions']
            if predictions:
                report.append(f"   {'':20s} Konfusionsmatrix:")
                categories = sorted(set(list(predictions.keys()) + 
                                      [p for preds in predictions.values() for p in preds.keys()]))
                
                # Header
                header = "   " + " " * 20 + "Vorhersage:"
                report.append(header)
                header_cats = "   " + " " * 20
                for cat in categories:
                    header_cats += f"{cat[:8]:>10s}"
                report.append(header_cats)
                
                # Zeilen
                for true_cat in categories:
                    row = f"   {'Wahr: ' + true_cat:20s}"
                    for pred_cat in categories:
                        count = predictions.get(true_cat, {}).get(pred_cat, 0)
                        if count > 0:
                            if true_cat == pred_cat:
                                row += f"{Fore.GREEN}{count:10d}{Fore.RESET}"
                            else:
                                row += f"{Fore.RED}{count:10d}{Fore.RESET}"
                        else:
                            row += f"{'':10s}"
                    report.append(row)
        
        # Kategorie-basierte Analyse
        report.append("\n\nKATEGORIE-ANALYSE:")
        category_stats = defaultdict(lambda: {'total': 0, 'correct_by_clf': defaultdict(int)})
        
        for entry in self.results_history:
            true_cat = entry['result'].get('true_category')
            if true_cat:
                category_stats[true_cat]['total'] += 1
                
                individual_scores = entry['result'].get('individual_scores', {})
                for clf_name, scores_data in individual_scores.items():
                    categories = scores_data.get('categories', {})
                    if categories:
                        predicted = max(categories, key=categories.get)
                        if predicted == true_cat:
                            category_stats[true_cat]['correct_by_clf'][clf_name] += 1
        
        for category, stats in sorted(category_stats.items()):
            report.append(f"\n   {category.upper()}:")
            report.append(f"   Gesamtanzahl: {stats['total']}")
            report.append(f"   Beste Klassifikatoren:")
            
            clf_accuracy = []
            for clf_name in self.classifier_performance.keys():
                correct = stats['correct_by_clf'].get(clf_name, 0)
                accuracy = correct / stats['total'] if stats['total'] > 0 else 0
                clf_accuracy.append((clf_name, accuracy, correct))
            
            clf_accuracy.sort(key=lambda x: x[1], reverse=True)
            
            for clf_name, accuracy, correct in clf_accuracy[:3]:  # Top 3
                report.append(f"      - {clf_name}: {accuracy*100:.1f}% ({correct}/{stats['total']})")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def generate_summary_report(self) -> str:
        """Generiert zusammenfassenden Report"""
        if not self.results_history:
            return "Keine Klassifikationen durchgeführt."
        
        # Sammle Statistiken
        total_classifications = len(self.results_history)
        category_counts = {}
        threat_levels = {}
        classifier_performance = {}
        channels = {}
        correct_predictions = 0
        total_with_truth = 0
        
        for entry in self.results_history:
            result = entry['result']
            
            # Kategorie-Zählung
            combined = result.get('combined_score', {})
            if combined:
                best_cat = max(combined, key=combined.get)
                category_counts[best_cat] = category_counts.get(best_cat, 0) + 1
                
                # Prüfe Korrektheit
                true_cat = result.get('true_category')
                if true_cat:
                    total_with_truth += 1
                    if best_cat == true_cat:
                        correct_predictions += 1
            
            # Bedrohungslevel
            threat = result.get('threat_analysis', {}).get('level', 'normal')
            threat_levels[threat] = threat_levels.get(threat, 0) + 1
            
            # Klassifikator-Performance
            for clf_name, data in result.get('individual_scores', {}).items():
                if clf_name not in classifier_performance:
                    classifier_performance[clf_name] = {
                        'total_time': 0,
                        'count': 0,
                        'correct': 0
                    }
                classifier_performance[clf_name]['total_time'] += data.get('processing_time', 0)
                classifier_performance[clf_name]['count'] += 1
            
            # Kanäle
            channel = result.get('event', {}).get('metadata', {}).get('channel', 'unknown')
            channels[channel] = channels.get(channel, 0) + 1
        
        # Erstelle Report
        report = []
        report.append("="*60)
        report.append("KLASSIFIKATIONS-ZUSAMMENFASSUNG")
        report.append("="*60)
        report.append(f"\nGesamte Klassifikationen: {total_classifications}")
        
        # Gesamtgenauigkeit wenn verfügbar
        if total_with_truth > 0:
            overall_accuracy = correct_predictions / total_with_truth
            report.append(f"Klassifikationen mit Ground Truth: {total_with_truth}")
            report.append(f"Gesamtgenauigkeit: {overall_accuracy*100:.1f}%")
        
        report.append("\nKATEGORIEN-VERTEILUNG:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_classifications) * 100
            report.append(f"   {cat}: {count} ({percentage:.1f}%)")
        
        report.append("\nBEDROHUNGSLEVEL:")
        for level, count in sorted(threat_levels.items()):
            percentage = (count / total_classifications) * 100
            report.append(f"   {level}: {count} ({percentage:.1f}%)")
        
        report.append("\nEINGANGSKANÄLE:")
        for channel, count in sorted(channels.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_classifications) * 100
            report.append(f"   {channel}: {count} ({percentage:.1f}%)")
        
        report.append("\nKLASSIFIKATOR-PERFORMANCE:")
        for clf_name, stats in classifier_performance.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            report.append(f"   {clf_name}:")
            report.append(f"      Durchschn. Zeit: {avg_time*1000:.1f}ms")
            report.append(f"      Verarbeitungen: {stats['count']}")
        
        # Performance-Report anhängen wenn aktiviert
        if self.performance_tracking_enabled:
            report.append(self.generate_performance_report())
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


class VisualizationHelper:
    """Hilfsklasse für visuelle Darstellungen"""
    
    @staticmethod
    def create_ascii_chart(data: Dict[str, float], title: str, width: int = 50) -> str:
        """Erstellt ASCII-Balkendiagramm"""
        if not data:
            return f"{title}\nKeine Daten vorhanden"
        
        max_value = max(data.values()) if data.values() else 1
        lines = [title, "-" * width]
        
        for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((value / max_value) * (width - len(label) - 10))
            bar = "█" * bar_length
            percentage = (value / sum(data.values())) * 100 if sum(data.values()) > 0 else 0
            lines.append(f"{label:15s} |{bar} {percentage:.1f}%")
        
        return "\n".join(lines)
    
    @staticmethod
    def create_threat_indicator(level: str) -> str:
        """Erstellt visuellen Bedrohungsindikator"""
        indicators = {
            'normal': "████░░░░░░ NORMAL",
            'warning': "███████░░░ WARNUNG",
            'critical': "█████████░ KRITISCH",
            'emergency': "██████████ NOTFALL"
        }
        return indicators.get(level, "░░░░░░░░░░ UNBEKANNT")
    
    @staticmethod
    def create_classifier_comparison_chart(performance_data: Dict[str, Dict]) -> str:
        """Erstellt Vergleichschart für Klassifikatoren"""
        lines = ["KLASSIFIKATOR-VERGLEICH", "=" * 60]
        
        for clf_name, data in sorted(performance_data.items(), 
                                    key=lambda x: x[1].get('accuracy', 0), 
                                    reverse=True):
            accuracy = data.get('accuracy', 0)
            bar_length = int(accuracy * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            lines.append(f"{clf_name:20s} [{bar}] {accuracy*100:.1f}%")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_time_delta(seconds: float) -> str:
        """Formatiert Zeitdifferenz human-readable"""
        if seconds < 0.001:
            return f"{seconds*1000000:.0f}μs"
        elif seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            return f"{seconds/60:.1f}min"


# Beispiel-Integration
def setup_enhanced_logging(enable_performance_tracking: bool = False):
    """Konfiguriert erweitertes Logging"""
    # Basis-Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('classifier_system.log', encoding='utf-8')
        ]
    )
    
    # Erstelle spezialisierten Logger
    classification_logger = ClassificationLogger()
    
    if enable_performance_tracking:
        classification_logger.enable_performance_tracking()
    
    return classification_logger


# Demo-Funktion
def demo_enhanced_logging():
    """Demonstriert erweiterte Logging-Funktionen"""
    logger = ClassificationLogger()
    logger.enable_performance_tracking()
    viz = VisualizationHelper()
    
    # Simuliere Klassifikationsergebnis mit Ground Truth
    demo_result = {
        'event': {
            'message': 'Aufzugstür klemmt im 3. Stock - DRINGEND!',
            'metadata': {
                'channel': 'Email',
                'priority': 2
            }
        },
        'combined_score': {
            'fahrkabine': 0.85,
            'seil': 0.10,
            'aufzugsgetriebe': 0.05
        },
        'individual_scores': {
            'SimpleEmbedding': {
                'categories': {'fahrkabine': 0.8, 'seil': 0.1, 'aufzugsgetriebe': 0.1},
                'confidence': 0.8,
                'processing_time': 0.002
            },
            'EnhancedRuleBased': {
                'categories': {'fahrkabine': 0.95, 'seil': 0.03, 'aufzugsgetriebe': 0.02},
                'confidence': 0.95,
                'processing_time': 0.001
            },
            'TfidfML': {
                'categories': {'fahrkabine': 0.7, 'seil': 0.2, 'aufzugsgetriebe': 0.1},
                'confidence': 0.7,
                'processing_time': 0.003
            }
        },
        'threat_analysis': {
            'level': 'critical',
            'indicators': ['dringend', 'klemmt'],
            'confidence': 0.9
        },
        'system_status': {
            'mode': 'fahrkabine_focus',
            'threat_level': 'critical',
            'recent_warnings': {
                'fahrkabine': 3,
                'seil': 0,
                'aufzugsgetriebe': 1
            }
        },
        'processing_time': 0.853,
        'classifier_count': 3,
        'true_category': 'fahrkabine',  # Ground Truth
        'final_prediction': 'fahrkabine'
    }
    
    # Logge Ergebnis
    logger.log_classification_result(demo_result)
    
    # Simuliere weitere Ergebnisse für Performance-Analyse
    test_results = [
        {'true': 'seil', 'predicted': 'seil'},
        {'true': 'fahrkabine', 'predicted': 'fahrkabine'},
        {'true': 'aufzugsgetriebe', 'predicted': 'seil'},
        {'true': 'fahrkabine', 'predicted': 'fahrkabine'},
        {'true': 'seil', 'predicted': 'aufzugsgetriebe'}
    ]
    
    for test in test_results:
        result = demo_result.copy()
        result['true_category'] = test['true']
        result['combined_score'] = {test['predicted']: 0.7, 'other': 0.3}
        result['final_prediction'] = test['predicted']
        logger.log_classification_result(result)
    
    # Zeige Performance-Report
    print(logger.generate_performance_report())
    
    # Zeige Summary mit Performance
    print("\n" + logger.generate_summary_report())


if __name__ == "__main__":
    demo_enhanced_logging()