# enhanced_logging.py - Verbessertes Logging und Visualisierung
import logging
from typing import Dict, List, Any
from datetime import datetime
from colorama import Fore, Back, Style, init
import json

# Initialisiere Colorama für Windows-Kompatibilität
init(autoreset=True)

class ClassificationLogger:
    """Spezieller Logger für Klassifikationsergebnisse"""
    
    def __init__(self, name: str = "ClassificationLogger"):
        self.logger = logging.getLogger(name)
        self.results_history = []
    
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
        
        # Finde beste Kategorie
        best_category = max(combined_scores, key=combined_scores.get) if combined_scores else 'unknown'
        best_score = combined_scores.get(best_category, 0)
        
        # Formatiere Ausgabe
        print("\n" + "="*80)
        print(f"{Fore.CYAN}📋 KLASSIFIKATIONSERGEBNIS")
        print("="*80)
        
        # Nachricht
        print(f"\n{Fore.WHITE}Nachricht: {Style.BRIGHT}{message}")
        
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
        print(f"\n{Fore.GREEN}✅ FINALE KLASSIFIKATION: {color}{Style.BRIGHT}{best_category.upper()}{Style.RESET_ALL} ({best_score*100:.1f}%)")
        
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
            
            print(f"\n{Fore.YELLOW}⚠️  BEDROHUNGSANALYSE:")
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
            
            print(f"\n{Fore.CYAN}🎯 SYSTEM-STATUS:")
            print(f"   Modus: {mode_color}{mode.upper()}{Style.RESET_ALL}")
            
            warnings = system_status.get('recent_warnings', {})
            if warnings:
                print(f"   Aktuelle Warnungen:")
                for category, count in warnings.items():
                    if count > 0:
                        print(f"      - {category}: {count} Warnungen")
        
        # Detaillierte Klassifikator-Ergebnisse
        print(f"\n{Fore.YELLOW}🔍 EINZELNE KLASSIFIKATOR-ERGEBNISSE:")
        
        for classifier_name, scores_data in individual_scores.items():
            categories = scores_data.get('categories', {})
            confidence = scores_data.get('confidence', 0)
            proc_time = scores_data.get('processing_time', 0)
            
            # Finde beste Kategorie für diesen Klassifikator
            if categories:
                best_cat = max(categories, key=categories.get)
                best_score = categories[best_cat]
                
                # Formatiere Klassifikator-Name
                classifier_display = {
                    'SimpleEmbedding': '📝 Einfach',
                    'EnhancedRuleBased': '📏 Regel-basiert',
                    'OllamaLLM': '🤖 LLM (Ollama)'
                }.get(classifier_name, classifier_name)
                
                print(f"\n   {classifier_display}:")
                print(f"      Wahl: {category_colors.get(best_cat, '')}{best_cat}{Fore.RESET} ({best_score*100:.1f}%)")
                print(f"      Konfidenz: {confidence*100:.1f}%")
                print(f"      Zeit: {proc_time*1000:.1f}ms")
                
                # Zeige Top-3 Scores
                sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                scores_str = " | ".join([f"{cat}:{score*100:.0f}%" for cat, score in sorted_cats])
                print(f"      Scores: {scores_str}")
        
        # Kombinierte Scores
        print(f"\n{Fore.GREEN}📊 KOMBINIERTE SCORES:")
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        for cat, score in sorted_combined:
            if score > 0.1:  # Nur relevante Scores
                bar_length = int(score * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                color = category_colors.get(cat, Fore.WHITE)
                print(f"   {color}{cat:15s}{Fore.RESET} [{bar}] {score*100:5.1f}%")
        
        # Performance
        total_time = result.get('processing_time', 0)
        print(f"\n{Fore.CYAN}⏱️  PERFORMANCE:")
        print(f"   Gesamtzeit: {total_time*1000:.1f}ms")
        print(f"   Klassifikatoren: {result.get('classifier_count', 0)}")
        
        print("\n" + "="*80)
    
    def log_mode_change(self, old_mode: str, new_mode: str, reason: str):
        """Loggt Modus-Änderungen"""
        mode_emojis = {
            'normal': '🟢',
            'fahrkabine_focus': '🔵',
            'seil_focus': '🟣',
            'getriebe_focus': '🔷',
            'emergency_all': '🔴'
        }
        
        old_emoji = mode_emojis.get(old_mode, '⚪')
        new_emoji = mode_emojis.get(new_mode, '⚪')
        
        print(f"\n{Back.YELLOW}{Fore.BLACK} MODUS-ÄNDERUNG {Style.RESET_ALL}")
        print(f"{old_emoji} {old_mode} → {new_emoji} {new_mode}")
        print(f"Grund: {reason}\n")
    
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
        
        for entry in self.results_history:
            result = entry['result']
            
            # Kategorie-Zählung
            combined = result.get('combined_score', {})
            if combined:
                best_cat = max(combined, key=combined.get)
                category_counts[best_cat] = category_counts.get(best_cat, 0) + 1
            
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
        report.append("📊 KLASSIFIKATIONS-ZUSAMMENFASSUNG")
        report.append("="*60)
        report.append(f"\nGesamte Klassifikationen: {total_classifications}")
        
        report.append("\n🎯 KATEGORIEN-VERTEILUNG:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_classifications) * 100
            report.append(f"   {cat}: {count} ({percentage:.1f}%)")
        
        report.append("\n⚠️  BEDROHUNGSLEVEL:")
        for level, count in sorted(threat_levels.items()):
            percentage = (count / total_classifications) * 100
            report.append(f"   {level}: {count} ({percentage:.1f}%)")
        
        report.append("\n📡 EINGANGSKANÄLE:")
        for channel, count in sorted(channels.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_classifications) * 100
            report.append(f"   {channel}: {count} ({percentage:.1f}%)")
        
        report.append("\n⚡ KLASSIFIKATOR-PERFORMANCE:")
        for clf_name, stats in classifier_performance.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            report.append(f"   {clf_name}:")
            report.append(f"      Durchschn. Zeit: {avg_time*1000:.1f}ms")
            report.append(f"      Verarbeitungen: {stats['count']}")
        
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
            'normal': "🟢 ████░░░░░░ NORMAL",
            'warning': "🟡 ███████░░░ WARNUNG",
            'critical': "🟠 █████████░ KRITISCH",
            'emergency': "🔴 ██████████ NOTFALL"
        }
        return indicators.get(level, "⚪ ░░░░░░░░░░ UNBEKANNT")
    
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
def setup_enhanced_logging():
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
    
    return classification_logger


# Demo-Funktion
def demo_enhanced_logging():
    """Demonstriert erweiterte Logging-Funktionen"""
    logger = ClassificationLogger()
    viz = VisualizationHelper()
    
    # Simuliere Klassifikationsergebnis
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
            'OllamaLLM': {
                'categories': {'fahrkabine': 0.8, 'seil': 0.15, 'aufzugsgetriebe': 0.05},
                'confidence': 0.75,
                'processing_time': 0.850
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
        'classifier_count': 3
    }
    
    # Logge Ergebnis
    logger.log_classification_result(demo_result)
    
    # Zeige Visualisierungen
    print("\n" + viz.create_threat_indicator('critical'))
    
    # Chart
    chart_data = {'fahrkabine': 0.85, 'seil': 0.10, 'aufzugsgetriebe': 0.05}
    print("\n" + viz.create_ascii_chart(chart_data, "Klassifikations-Verteilung"))
    
    # Modus-Änderung
    logger.log_mode_change('normal', 'fahrkabine_focus', 'Kritische Türstörung erkannt')
    
    # Summary
    print("\n" + logger.generate_summary_report())


if __name__ == "__main__":
    demo_enhanced_logging()