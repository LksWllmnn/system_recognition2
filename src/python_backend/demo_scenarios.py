# demo_scenarios.py - Realistische Demo-Szenarien
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio
import random

class DemoScenarioGenerator:
    """Generiert realistische Aufzugs-Szenarien"""
    
    def __init__(self):
        self.scenarios = {
            'normal_maintenance': self._create_normal_scenarios(),
            'warning_escalation': self._create_warning_escalation(),
            'emergency_situation': self._create_emergency_scenarios(),
            'multi_channel_test': self._create_multi_channel_scenarios()
        }
    
    def _create_normal_scenarios(self) -> List[Dict[str, Any]]:
        """Normale Wartungsmeldungen"""
        return [
            {
                'channel': 'DirectInput',
                'message': 'Monatliche Inspektion durchgeführt - alle Systeme normal',
                'metadata': {'technician_id': 'TECH_042', 'routine': True},
                'expected_category': 'unknown',
                'expected_threat': 'normal'
            },
            {
                'channel': 'Email',
                'message': 'Betreff: Wartungsprotokoll\nInhalt: Schmierung der Führungsschienen abgeschlossen',
                'metadata': {'from': 'wartung@aufzug-service.de'},
                'expected_category': 'aufzugsgetriebe',
                'expected_threat': 'normal'
            },
            {
                'channel': 'SMS',
                'message': 'Fahrkabine gereinigt, Beleuchtung getauscht',
                'metadata': {'from': '+49171234567'},
                'expected_category': 'fahrkabine',
                'expected_threat': 'normal'
            }
        ]
    
    def _create_warning_escalation(self) -> List[Dict[str, Any]]:
        """Eskalierendes Warnung-Szenario"""
        base_time = datetime.now()
        return [
            # Erste Warnung
            {
                'channel': 'Phone',
                'message': 'Die Tür im zweiten Stock schließt etwas langsamer als normal',
                'metadata': {'duration': 45, 'caller': '+49301234567'},
                'timestamp': base_time,
                'expected_category': 'fahrkabine',
                'expected_threat': 'warning'
            },
            # Zweite Warnung (10 Minuten später)
            {
                'channel': 'Email',
                'message': 'Betreff: Türproblem\nMehrere Mitarbeiter berichten verzögerte Türschließung Stock 2',
                'metadata': {'priority_flag': True},
                'timestamp': base_time + timedelta(minutes=10),
                'expected_category': 'fahrkabine',
                'expected_threat': 'warning'
            },
            # Dritte Warnung (15 Minuten später) - sollte Fokus-Modus auslösen
            {
                'channel': 'SMS',
                'message': 'Tür Stock 2 klemmt jetzt häufiger - bitte prüfen!',
                'metadata': {'from': 'Hausmeister'},
                'timestamp': base_time + timedelta(minutes=15),
                'expected_category': 'fahrkabine',
                'expected_threat': 'warning',
                'expected_mode_change': 'fahrkabine_focus'
            },
            # Kritische Meldung (20 Minuten später)
            {
                'channel': 'DirectInput',
                'message': 'DRINGEND: Tür blockiert komplett - Aufzug außer Betrieb genommen',
                'metadata': {'severity': 'high'},
                'timestamp': base_time + timedelta(minutes=20),
                'expected_category': 'fahrkabine',
                'expected_threat': 'critical'
            }
        ]
    
    def _create_emergency_scenarios(self) -> List[Dict[str, Any]]:
        """Notfall-Szenarien"""
        return [
            {
                'channel': 'EmergencyButton',
                'message': 'NOTFALL: Notfallknopf Aufzug 3 Stock 7 aktiviert',
                'metadata': {
                    'button_id': 'BTN_03_07',
                    'location': 'Hauptgebäude',
                    'auto_priority': 2
                },
                'expected_category': 'unknown',
                'expected_threat': 'emergency',
                'expected_mode_change': 'emergency_all'
            },
            {
                'channel': 'Phone',
                'message': 'Notfall! Person im Aufzug eingeschlossen, Aufzug steht zwischen Stock 4 und 5',
                'metadata': {'emergency_call': True, 'duration': 120},
                'expected_category': 'fahrkabine',
                'expected_threat': 'emergency'
            },
            {
                'channel': 'SMS',
                'message': 'Seil gerissen!!! Aufzug 2 sofort stilllegen!!!',
                'metadata': {'sender_verified': True},
                'expected_category': 'seil',
                'expected_threat': 'emergency'
            }
        ]
    
    def _create_multi_channel_scenarios(self) -> List[Dict[str, Any]]:
        """Test verschiedener Kanäle für gleiche Probleme"""
        return [
            # Getriebe-Problem über verschiedene Kanäle
            {
                'channel': 'Phone',
                'message': 'Äh hallo, der Aufzug macht so komische brummende Geräusche',
                'metadata': {'transcript_confidence': 0.85},
                'expected_category': 'aufzugsgetriebe',
                'expected_threat': 'warning'
            },
            {
                'channel': 'Email',
                'message': 'Betreff: Ungewöhnliche Vibrationen\nSeit heute morgen vibriert Aufzug 1 stark',
                'metadata': {'attachments': ['video_vibration.mp4']},
                'expected_category': 'aufzugsgetriebe',
                'expected_threat': 'warning'
            },
            {
                'channel': 'DirectInput',
                'message': 'Getriebegeräusch Aufzug 1 - Lager prüfen',
                'metadata': {'source': 'maintenance_app'},
                'expected_category': 'aufzugsgetriebe',
                'expected_threat': 'warning'
            },
            # Seil-Inspektion
            {
                'channel': 'DirectInput',
                'message': 'Routineinspektion: Leichte Abnutzung am Tragseil erkennbar',
                'metadata': {'inspection_type': 'visual'},
                'expected_category': 'seil',
                'expected_threat': 'normal'
            },
            {
                'channel': 'Email',
                'message': 'Seilspannung nachjustiert nach Messung',
                'metadata': {'measurement_included': True},
                'expected_category': 'seil',
                'expected_threat': 'normal'
            }
        ]
    
    def generate_scenario_batch(self, scenario_type: str) -> List[Dict[str, Any]]:
        """Generiert eine Batch von Szenarien"""
        if scenario_type not in self.scenarios:
            raise ValueError(f"Unbekannter Szenario-Typ: {scenario_type}")
        
        return self.scenarios[scenario_type]
    
    def create_performance_test_batch(self, count: int = 100) -> List[Dict[str, Any]]:
        """Erstellt große Menge von Test-Nachrichten für Performance-Tests"""
        import random
        
        channels = ['SMS', 'Email', 'Phone', 'DirectInput']
        
        # Nachrichtentemplates
        templates = {
            'fahrkabine': [
                'Tür {} verzögert',
                'Bedienfeld in Stock {} reagiert nicht',
                'Beleuchtung {} ausgefallen',
                'Kabine {} macht Geräusche',
                'Notsprechanlage {} defekt'
            ],
            'seil': [
                'Seilspannung {} zu niedrig',
                'Tragseil {} zeigt Verschleiß',
                'Seilrolle {} quietscht',
                'Führungsseil {} verschoben',
                'Seilüberwachung {} meldet Fehler'
            ],
            'aufzugsgetriebe': [
                'Motor {} vibriert stark',
                'Getriebe {} verliert Öl',
                'Temperatur {} zu hoch',
                'Schmierung {} erforderlich',
                'Steuerung {} fehlerhaft'
            ]
        }
        
        messages = []
        for i in range(count):
            category = random.choice(list(templates.keys()))
            template = random.choice(templates[category])
            channel = random.choice(channels)
            
            # Variiere Bedrohungslevel
            threat_modifier = random.random()
            if threat_modifier < 0.7:
                threat_words = ['leicht', 'minimal', 'gering']
                expected_threat = 'normal'
            elif threat_modifier < 0.9:
                threat_words = ['stark', 'deutlich', 'erheblich']
                expected_threat = 'warning'
            else:
                threat_words = ['kritisch', 'sofort', 'dringend']
                expected_threat = 'critical'
            
            message_text = template.format(random.choice(threat_words))
            
            messages.append({
                'channel': channel,
                'message': message_text,
                'metadata': {
                    'test_id': f'PERF_{i:04d}',
                    'category_hint': category
                },
                'expected_category': category,
                'expected_threat': expected_threat
            })
        
        return messages
    
    def create_stress_test_scenario(self) -> List[Dict[str, Any]]:
        """Erstellt Stress-Test mit simultanen kritischen Meldungen"""
        base_time = datetime.now()
        return [
            {
                'channel': 'EmergencyButton',
                'message': 'NOTFALL: Notknopf Aufzug 1',
                'timestamp': base_time,
                'expected_threat': 'emergency'
            },
            {
                'channel': 'EmergencyButton',
                'message': 'NOTFALL: Notknopf Aufzug 2',
                'timestamp': base_time + timedelta(seconds=2),
                'expected_threat': 'emergency'
            },
            {
                'channel': 'Phone',
                'message': 'Alle Aufzüge ausgefallen! Stromausfall!',
                'timestamp': base_time + timedelta(seconds=5),
                'expected_threat': 'emergency'
            },
            {
                'channel': 'SMS',
                'message': 'Evakuierung eingeleitet - alle Aufzüge gesperrt',
                'timestamp': base_time + timedelta(seconds=10),
                'expected_threat': 'emergency'
            }
        ]


# Demo-Runner für Szenarien
async def run_scenario_demo(classifier_system, scenario_type: str = 'warning_escalation'):
    """Führt Demo-Szenario aus"""
    from enhanced_logging import ClassificationLogger
    from input_channels import MultiChannelInputHandler
    
    # Setup
    generator = DemoScenarioGenerator()
    logger = ClassificationLogger()
    input_handler = MultiChannelInputHandler()
    
    # Initialisiere
    await input_handler.initialize_all_channels()
    
    print(f"\n🎬 STARTE SZENARIO: {scenario_type.upper()}")
    print("="*70)
    
    # Hole Szenario
    scenarios = generator.generate_scenario_batch(scenario_type)
    
    for i, scenario in enumerate(scenarios):
        print(f"\n📨 Nachricht {i+1}/{len(scenarios)}")
        
        # Simuliere Kanal-Input
        channel_data = {
            'text': scenario['message'],
            'metadata': scenario.get('metadata', {}),
            'timestamp': scenario.get('timestamp', datetime.now())
        }
        
        # Verarbeite durch Input-Handler
        input_message = await input_handler.process_channel_input(
            scenario['channel'], 
            channel_data
        )
        
        if input_message:
            # Erstelle Event
            from event import Event
            event = Event(
                timestamp=input_message.timestamp,
                message=input_message.processed_content,
                metadata={
                    'channel': input_message.channel,
                    'priority': input_message.priority,
                    **input_message.metadata
                }
            )
            
            # Klassifiziere
            result = await classifier_system.classify_event(event)
            
            # Logge Ergebnis
            logger.log_classification_result(result)
            
            # Validiere Erwartungen
            if 'expected_category' in scenario:
                combined = result['combined_score']
                actual_category = max(combined, key=combined.get)
                if actual_category == scenario['expected_category']:
                    print(f"✅ Kategorie korrekt: {actual_category}")
                else:
                    print(f"❌ Kategorie falsch: Erwartet {scenario['expected_category']}, erhalten {actual_category}")
            
            if 'expected_threat' in scenario:
                actual_threat = result['threat_analysis']['level']
                if actual_threat == scenario['expected_threat']:
                    print(f"✅ Bedrohungslevel korrekt: {actual_threat}")
                else:
                    print(f"❌ Bedrohungslevel falsch: Erwartet {scenario['expected_threat']}, erhalten {actual_threat}")
            
            # Pause zwischen Nachrichten
            await asyncio.sleep(1)
    
    # Zeige Zusammenfassung
    print("\n" + logger.generate_summary_report())


if __name__ == "__main__":
    import asyncio
    
    # Demo verschiedene Szenarien
    async def main():
        # Dummy classifier system für Demo
        class DummyClassifier:
            async def classify_event(self, event):
                return {
                    'event': event.to_dict(),
                    'combined_score': {'fahrkabine': 0.8, 'seil': 0.1, 'aufzugsgetriebe': 0.1},
                    'individual_scores': {},
                    'threat_analysis': {'level': 'warning', 'indicators': [], 'confidence': 0.5},
                    'system_status': {'mode': 'normal', 'threat_level': 'warning', 'recent_warnings': {}},
                    'processing_time': 0.1,
                    'classifier_count': 3
                }
        
        classifier = DummyClassifier()
        
        # Teste verschiedene Szenarien
        for scenario in ['normal_maintenance', 'warning_escalation', 'emergency_situation']:
            await run_scenario_demo(classifier, scenario)
            print("\n" + "="*70 + "\n")
    
    asyncio.run(main())