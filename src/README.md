# Socket-basiertes Multi-Classifier System

## 🏗️ Neue Architektur - Saubere Trennung

```
src/
├── Python Backend (Klassifikation)
│   ├── event.py                     # Event Datenstrukturen
│   ├── base_classifier.py           # Basis Klassifikator Interface
│   ├── simple_classifier.py         # Einfacher Embedding Klassifikator
│   ├── rule_classifier.py           # Regel-basierter Klassifikator
│   ├── ollama_classifier.py         # LLM Klassifikator
│   ├── multi_classifier_system.py   # Multi-Klassifikator System
│   └── socket_server.py             # Socket Server
│
├── TypeScript Frontend (Demo)
│   ├── socket_client.ts             # Socket Client
│   ├── demo_system.ts               # Demo System
│   └── socket_demo.ts               # Haupt-Demo (Entry Point)
│
└── Data
    └── messages.json                # (Optional) Demo-Nachrichten
```

## 🎯 Verbesserungen

### ✅ **Modulare Architektur**
- Jede Klasse in eigener Datei
- Klare Interfaces und Abhängigkeiten
- Einfache Erweiterbarkeit

### ✅ **Saubere Trennung**
- **Python**: Klassifikations-Engine
- **TypeScript**: Demo und Client
- **Socket**: Kommunikationsprotokoll

### ✅ **Robuste Kommunikation**
- Message Framing für zuverlässige Übertragung
- Timeout-Handling und Reconnection
- Detaillierte Fehler-Diagnose

### ✅ **Besseres Error Handling**
- Spezifische Fehlererkennung
- Graceful Degradation
- Detailliertes Logging

## 🚀 Installation und Ausführung

### 1. Python Dependencies
```bash
pip install requests
```

### 2. TypeScript Dependencies
```bash
npm install
```

### 3. Demo starten
```bash
npx ts-node src/demo_system.ts
```

### 4. Custom Port
```bash
npx ts-node src/demo_system.ts --port 9999
```

## 🔧 Verwendung

### Einzelne Module testen

**Python Server direkt:**
```bash
cd src
python socket_server.py --log-level DEBUG
```

**Client separat verwenden:**
```typescript
import { SocketClassifierClient } from './socket_client';

const client = new SocketClassifierClient();
await client.connect();
const result = await client.classify("Hallo, wie geht es?");
console.log(result);
```

## 📊 Features

### **Multi-Klassifikator System**
- **SimpleEmbedding**: Keyword-basiert, schnell
- **EnhancedRuleBased**: Regex-Regeln, präzise
- **OllamaLLM**: KI-basiert, intelligent (optional)

### **Parallele Verarbeitung**
- Alle Klassifikatoren laufen gleichzeitig
- Ergebnisse werden intelligent kombiniert
- Fallback bei Fehlern einzelner Klassifikatoren

### **Performance Monitoring**
- Verarbeitungszeiten pro Klassifikator
- System-Statistiken
- Request-Tracking

## 🔍 Debugging

### Debug-Mode aktivieren
```bash
# Python Server mit Debug-Logs
python socket_server.py --log-level DEBUG

# TypeScript mit erweiterten Logs
DEBUG=* npx ts-node src/demo_system.ts
```

### Häufige Probleme

**1. Port bereits belegt:**
```bash
# Windows
netstat -an | findstr "8888"

# Linux/Mac
netstat -an | grep 8888

# Lösung: Anderen Port verwenden
npx ts-node src/demo_system.ts --port 9999
```

**2. Python Dependencies fehlen:**
```bash
pip install requests
```

**3. Ollama nicht verfügbar:**
- System läuft trotzdem mit anderen Klassifikatoren
- Ollama optional installieren: `curl -fsSL https://ollama.ai/install.sh | sh`

## 🎮 Demo-Features

### **Interaktive Demo**
- 10 vordefinierte Test-Nachrichten
- Verschiedene Kategorien (greeting, question, problem, etc.)
- Echtzeit-Klassifikation mit Timing

### **Detaillierte Ausgabe**
```
[1/10] "Hallo! Wie geht es dir heute?"
  🎯 Kategorie: greeting (85.3%)
  ✅ Erwartet: greeting
  ⏱️ Zeit: 23.7ms
  🔧 Klassifikatoren: 3
  📊 Scores: greeting:85%, positive:12%, unknown:3%
```

### **Performance-Statistiken**
```
📊 FINALE STATISTIKEN:
📈 Verarbeitete Events: 10
⏱️ Durchschnittliche Zeit: 0.156s
🔧 Aktive Klassifikatoren: 3
```

## 🛠️ Erweiterung

### Neuen Klassifikator hinzufügen

1. **Erstelle neue Datei**: `src/my_classifier.py`
```python
from base_classifier import BaseClassifier
from event import Event, ClassificationResult

class MyClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("MyClassifier")
    
    async def initialize(self) -> bool:
        # Initialisierung
        return True
    
    async def classify(self, event: Event) -> ClassificationResult:
        # Deine Klassifikations-Logik
        return ClassificationResult(...)
```

2. **Registriere in System**: `multi_classifier_system.py`
```python
from my_classifier import MyClassifier

# In __init__:
self.classifiers.append(MyClassifier())
```

### Custom Demo-Nachrichten

Erstelle `src/messages.json`:
```json
[
    {
        "text": "Meine Test-Nachricht",
        "expected_category": "custom",
        "metadata": {"priority": "high"}
    }
]
```

## 🔧 Konfiguration

### Server-Konfiguration
```python
# socket_server.py anpassen
server = SocketClassifierServer(
    host='0.0.0.0',  # Alle Interfaces
    port=8888
)
```

### Client-Konfiguration
```typescript
const client = new SocketClassifierClient({
    host: 'localhost',
    port: 8888,
    timeout: 10000,  // 10s Timeout
    reconnect: {
        enabled: true,
        max_attempts: 5,
        delay_ms: 1000,
        backoff_factor: 2.0
    }
});
```

## 📈 Roadmap

- [ ] Web-Interface für Live-Demo
- [ ] REST API zusätzlich zu Socket
- [ ] Mehr Klassifikatoren (BERT, spaCy)
- [ ] Konfigurierbare Klassifikations-Kategorien
- [ ] Performance-Dashboard
- [ ] Docker-Container

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle Feature-Branch
3. Implementiere saubere Module
4. Teste gründlich
5. Erstelle Pull Request

## 📝 Lizenz

MIT License - Siehe LICENSE Datei für Details.