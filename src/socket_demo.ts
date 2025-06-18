// socket_demo.ts - Demo mit Socket-basierter Kommunikation
import { SocketEnhancedSystemRecognizer, Event } from './socket_classifier_client';
import { spawn } from 'child_process';
import * as fs from 'fs';

interface OriginalMessage {
    typ: string;
    data: string;
    meldungseingang: number;
}

class SocketDemoSystem {
    private system: SocketEnhancedSystemRecognizer;
    private originalMessages: OriginalMessage[] = [];
    private pythonServer: any = null;

    constructor() {
        this.system = new SocketEnhancedSystemRecognizer();
    }

    async startPythonServer(): Promise<void> {
        return new Promise((resolve, reject) => {
            console.log('🚀 Starte Python Socket-Server...');
            
            this.pythonServer = spawn('python', ['socket_classifier_server.py'], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: './src',
                shell: true
            });

            // Server Output
            this.pythonServer.stdout.on('data', (data: Buffer) => {
                console.log(`🐍 Server: ${data.toString().trim()}`);
            });

            this.pythonServer.stderr.on('data', (data: Buffer) => {
                const message = data.toString().trim();
                console.log(`🐛 Server Debug: ${message}`);
                
                if (message.includes('Multi-Classifier System bereit')) {
                    // Warte kurz, dann versuche Verbindung
                    setTimeout(resolve, 1000);
                }
            });

            this.pythonServer.on('error', (error: Error) => {
                console.error('❌ Python Server Fehler:', error);
                reject(error);
            });

            // Timeout für Server-Start
            setTimeout(() => {
                reject(new Error('Python Server Start Timeout'));
            }, 10000);
        });
    }

    async initialize(): Promise<void> {
        // Lade ursprüngliche Nachrichten
        try {
            const possiblePaths = [
                './old/messages.json',
                '../old/messages.json',
                './messages.json',
                '../messages.json'
            ];

            let messageData = '';
            for (const path of possiblePaths) {
                if (fs.existsSync(path)) {
                    messageData = fs.readFileSync(path, 'utf-8');
                    console.log(`📂 Nachrichten geladen aus: ${path}`);
                    break;
                }
            }

            if (messageData) {
                this.originalMessages = JSON.parse(messageData);
                console.log(`📊 ${this.originalMessages.length} ursprüngliche Nachrichten geladen`);
            } else {
                console.log('⚠️ messages.json nicht gefunden, verwende Demo-Daten');
                this.createDemoMessages();
            }
        } catch (error) {
            console.error('Fehler beim Laden der messages.json:', error);
            this.createDemoMessages();
        }

        // Starte Python Server
        await this.startPythonServer();
        
        // Verbinde Client
        await this.system.initialize();
    }

    private createDemoMessages(): void {
        this.originalMessages = [
            {"typ": "warnung", "data": "Leichte Verzögerung bei Türöffnung festgestellt", "meldungseingang": 15},
            {"typ": "warnung", "data": "Schmierölstand leicht reduziert – Wartung empfohlen", "meldungseingang": 35},
            {"typ": "warnung", "data": "Vibrationen leicht erhöht – Überprüfung empfohlen", "meldungseingang": 50},
            {"typ": "warnung", "data": "Temperatur im Schaltschrank leicht erhöht", "meldungseingang": 72},
            {"typ": "bedrohung", "data": "Plötzlicher Systemausfall – Notbetrieb aktiviert", "meldungseingang": 76},
            {"typ": "warnung", "data": "Getriebeölstand nahe Mindestwert", "meldungseingang": 85},
            {"typ": "warnung", "data": "Unregelmäßige Beschleunigung beim Startvorgang", "meldungseingang": 87},
            {"typ": "warnung", "data": "Tür schließt verzögert nach Gewichtserkennung", "meldungseingang": 89},
            {"typ": "warnung", "data": "Antrieb benötigt längere Initialisierung", "meldungseingang": 93},
            {"typ": "bedrohung", "data": "Kein Zugang zur Steuerungseinheit – manuelle Übersteuerung notwendig", "meldungseingang": 95}
        ];
        console.log('📝 Demo-Nachrichten erstellt');
    }

    private convertToEvent(msg: OriginalMessage, index: number): Event {
        const severityMap: Record<string, 'info' | 'warnung' | 'bedrohung'> = {
            'info': 'info',
            'warnung': 'warnung', 
            'bedrohung': 'bedrohung'
        };

        let channel: 'funk' | 'sms' | 'email' | 'sensor' | 'manual' | 'emergency' = 'sensor';
        if (msg.typ === 'bedrohung') {
            channel = 'emergency';
        } else if (msg.data.includes('Wartung') || msg.data.includes('empfohlen')) {
            channel = 'email';
        }

        return {
            id: `demo_${index + 1}`,
            timestamp: Date.now() / 1000 + msg.meldungseingang,
            channel,
            severity: severityMap[msg.typ] || 'info',
            rawText: msg.data,
            source: 'demo_data'
        };
    }

    async runDemo(): Promise<void> {
        console.log('\n🎬 Starte Socket-basiertes Demo mit Aufzug-Daten');
        console.log('='.repeat(80));

        const relevantMessages = this.originalMessages.filter(
            msg => msg.typ === 'warnung' || msg.typ === 'bedrohung'
        );

        console.log(`🎯 Verarbeite ${relevantMessages.length} relevante Nachrichten...\n`);

        const startTime = Date.now();
        let processedCount = 0;

        for (let i = 0; i < relevantMessages.length; i++) {
            const msg = relevantMessages[i];
            const event = this.convertToEvent(msg, i);

            console.log(`\n⏰ T+${msg.meldungseingang}s - Event ${i + 1}/${relevantMessages.length}`);
            console.log(`📨 Original: [${msg.typ.toUpperCase()}] ${msg.data}`);

            try {
                await this.system.processEvent(event);
                processedCount++;

                // Pause zwischen Events
                if (i < relevantMessages.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }

            } catch (error) {
                console.error(`❌ Fehler bei Event ${event.id}:`, error);
            }
        }

        // Finale Statistiken
        const endTime = Date.now();
        const processingTime = (endTime - startTime) / 1000;

        console.log('\n' + '='.repeat(80));
        console.log('📊 SOCKET-DEMO-ZUSAMMENFASSUNG:');
        console.log(`⏱️  Verarbeitungszeit: ${processingTime.toFixed(1)}s`);
        console.log(`✅ Verarbeitete Events: ${processedCount}/${relevantMessages.length}`);

        const finalStatus = await this.system.getSystemStatus();
        console.log(`🎯 Finaler Systemzustand: ${finalStatus.state}`);
        console.log(`🔍 Aktiver Fokusmodus: ${finalStatus.focusMode || 'Keiner'}`);

        console.log('\n📈 EVENT-KATEGORIEN:');
        Object.entries(finalStatus.eventCounts).forEach(([category, count]) => {
            console.log(`  ${category}: ${count} Warnungen`);
        });

        console.log('\n💾 BERT-TRAININGSDATEN:');
        const trainingStats = finalStatus.trainingStats;
        console.log(`📚 Gesamt: ${trainingStats.total} Einträge`);
        
        if (trainingStats.total > 0) {
            Object.entries(trainingStats.by_category).forEach(([category, count]) => {
                console.log(`  📋 ${category}: ${count} Beispiele`);
            });
        } else {
            console.log('⚠️  Keine Trainingsdaten gesammelt - alle Klassifikatoren waren uneinig');
        }
    }

    destroy(): void {
        this.system.destroy();
        
        if (this.pythonServer) {
            console.log('🛑 Beende Python Server...');
            this.pythonServer.kill();
        }
    }
}

async function runSocketDemo() {
    const demo = new SocketDemoSystem();
    
    try {
        console.log('🔌 SOCKET-BASIERTES MULTI-CLASSIFIER DEMO');
        console.log('='.repeat(80));
        console.log('🚀 Vorteile der Socket-Lösung:');
        console.log('   • Zuverlässige bidirektionale Kommunikation');
        console.log('   • Keine stdin/stdout Buffer-Probleme');
        console.log('   • Bessere Fehlerbehandlung und Timeouts');
        console.log('   • Server kann mehrere Clients bedienen');
        console.log('   • Einfache Health-Checks und Monitoring');
        console.log('='.repeat(80));

        await demo.initialize();
        await demo.runDemo();

    } catch (error) {
        console.error('❌ Socket-Demo fehlgeschlagen:', error);
    } finally {
        demo.destroy();
    }
}

if (require.main === module) {
    runSocketDemo().catch(console.error);
}

export { SocketDemoSystem };