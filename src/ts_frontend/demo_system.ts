// demo_system.ts - Sauberes Demo System
import { spawn, ChildProcess } from 'child_process';
import { existsSync, writeFileSync } from 'fs';
import * as path from 'path';
import { SocketClassifierClient } from './socket_client';

interface DemoConfig {
    server: {
        host: string;
        port: number;
        startup_timeout: number;
    };
    client: {
        timeout: number;
        max_connection_attempts: number;
    };
    demo: {
        message_count: number;
        delay_between_messages: number;
    };
}

// Hilfsfunktion für Deep Partial
type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

interface DemoMessage {
    text: string;
    expected_category?: string;
    metadata?: any;
}

export class DemoSystem {
    private config: DemoConfig;
    private pythonServer: ChildProcess | null = null;
    private client: SocketClassifierClient | null = null;
    private serverReady = false;

    constructor(config: DeepPartial<DemoConfig> = {}) {
        this.config = {
            server: {
                host: 'localhost',
                port: 8888,
                startup_timeout: 15000,
                ...config.server
            },
            client: {
                timeout: 5000,
                max_connection_attempts: 3,
                ...config.client
            },
            demo: {
                message_count: 10,
                delay_between_messages: 1000,
                ...config.demo
            }
        };
    }

    async runDemo(): Promise<void> {
        console.log('🎮 Starte Demo im erweiterten Modus...');
        
        try {
            // 1. Teste Python Setup
            await this.validatePythonSetup();
            
            // 2. Starte Python Server
            await this.startPythonServer();
            
            // 3. Verbinde Client
            await this.connectClient();
            
            // 4. Führe Demo durch
            await this.runClassificationDemo();
            
            console.log('✅ Demo erfolgreich abgeschlossen!');
            
        } catch (error) {
            console.error('❌ Demo fehlgeschlagen:', error);
            throw error;
        } finally {
            await this.cleanup();
        }
    }

    private async validatePythonSetup(): Promise<void> {
        console.log('🧪 Teste Python-Setup...');
        
        // Prüfe Python
        try {
            const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
            const { spawn } = require('child_process');
            
            const pythonTest = spawn(pythonCmd, ['--version'], { shell: true });
            
            await new Promise((resolve, reject) => {
                let output = '';
                
                pythonTest.stdout?.on('data', (data: Buffer) => {
                    output += data.toString();
                });
                
                pythonTest.on('close', (code: number | null) => {
                    if (code === 0) {
                        const version = output.trim();
                        console.log(`✅ Python gefunden: ${version}`);
                        resolve(void 0);
                    } else {
                        reject(new Error('Python nicht gefunden'));
                    }
                });
            });
        } catch (error) {
            throw new Error(`Python-Setup Fehler: ${error}`);
        }

        // Prüfe Server-Datei
        const serverPath = path.join('src', 'python_backend', 'socket_server.py');
        if (!existsSync(serverPath)) {
            throw new Error(`Server-Datei nicht gefunden: ${serverPath}`);
        }
        console.log('✅ Server-Datei gefunden');
        
        console.log('✅ Python-Setup OK');
    }

    private async startPythonServer(): Promise<void> {
        console.log('🚀 Starte Python Socket-Server...');
        
        return new Promise((resolve, reject) => {
            const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
            
            this.pythonServer = spawn(pythonCmd, [
                'socket_server.py',
                '--host', this.config.server.host,
                '--port', this.config.server.port.toString(),
                '--log-level', 'INFO'
            ], {
                cwd: 'src/python_backend',
                stdio: ['ignore', 'pipe', 'pipe']
            });

            let serverOutput = '';
            let errorOutput = '';

            // Server Ready Detection
            const checkServerReady = (message: string) => {
                if (message.includes('Classifier Server gestartet') || 
                    message.includes('Warte auf Verbindungen')) {
                    this.serverReady = true;
                    console.log('✅ Python-Server bereit');
                    setTimeout(resolve, 1000); // Kurze Pause
                }
            };

            // Output Monitoring
            this.pythonServer.stdout?.on('data', (data: Buffer) => {
                const message = data.toString().trim();
                serverOutput += message + '\n';
                console.log(`🐍 Server: ${message}`);
                checkServerReady(message);
            });

            this.pythonServer.stderr?.on('data', (data: Buffer) => {
                const message = data.toString().trim();
                errorOutput += message + '\n';
                console.log(`🔧 Server Log: ${message}`);
                checkServerReady(message);
                
                // Check für kritische Fehler
                if (message.includes('ModuleNotFoundError') || 
                    message.includes('ImportError') ||
                    message.includes('Address already in use')) {
                    reject(new Error(`Server Fehler: ${message}`));
                }
            });

            this.pythonServer.on('error', (error: Error) => {
                reject(error);
            });

            this.pythonServer.on('exit', (code, signal) => {
                if (!this.serverReady) {
                    reject(new Error(`Server beendet mit Code ${code}, Signal: ${signal}`));
                }
            });

            // Startup Timeout
            setTimeout(() => {
                if (!this.serverReady) {
                    reject(new Error(`Server Start Timeout nach ${this.config.server.startup_timeout}ms`));
                }
            }, this.config.server.startup_timeout);
        });
    }

    private async connectClient(): Promise<void> {
        console.log('🔌 Verbinde Client...');
        
        this.client = new SocketClassifierClient({
            host: this.config.server.host,
            port: this.config.server.port,
            timeout: this.config.client.timeout,
            reconnect: {
                enabled: true,
                max_attempts: this.config.client.max_connection_attempts,
                delay_ms: 2000,
                backoff_factor: 1.5
            }
        });

        let attempts = 0;
        const maxAttempts = this.config.client.max_connection_attempts;

        while (attempts < maxAttempts) {
            attempts++;
            console.log(`🔌 Verbindungsversuch ${attempts}/${maxAttempts}...`);
            
            try {
                await this.client.connect();
                
                // Test mit Health Check
                const health = await this.client.healthCheck();
                console.log('✅ Client erfolgreich verbunden');
                console.log(`📊 Server Status: ${health.status}, Uptime: ${health.uptime_seconds?.toFixed(1)}s`);
                return;
                
            } catch (error) {
                console.error(`❌ Verbindungsversuch ${attempts} fehlgeschlagen:`, error);
                
                if (attempts < maxAttempts) {
                    console.log('⏳ Warte 2s vor nächstem Versuch...');
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
        }
        
        throw new Error(`Alle ${maxAttempts} Verbindungsversuche fehlgeschlagen`);
    }

    private async runClassificationDemo(): Promise<void> {
        console.log('🎯 Starte Klassifikations-Demo...');
        
        if (!this.client) {
            throw new Error('Client nicht verbunden');
        }

        // Erstelle Demo-Nachrichten
        const demoMessages = this.createDemoMessages();
        
        console.log(`📝 Klassifiziere ${demoMessages.length} Demo-Nachrichten...\n`);

        for (let i = 0; i < demoMessages.length; i++) {
            const msg = demoMessages[i];
            
            try {
                console.log(`[${i + 1}/${demoMessages.length}] "${msg.text}"`);
                
                const result = await this.client.classify(msg.text, msg.metadata);
                
                // Zeige Ergebnisse
                this.displayClassificationResult(result, msg.expected_category);
                
                // Kurze Pause zwischen Nachrichten
                if (i < demoMessages.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, this.config.demo.delay_between_messages));
                }
                
            } catch (error) {
                console.error(`❌ Fehler bei Nachricht ${i + 1}:`, error);
            }
        }

        // Zeige finale Statistiken
        try {
            const stats = await this.client.getStats();
            console.log('\n📊 FINALE STATISTIKEN:');
            console.log(`📈 Verarbeitete Events: ${stats.stats.total_events}`);
            console.log(`⏱️  Durchschnittliche Zeit: ${stats.stats.average_time?.toFixed(3)}s`);
            console.log(`🔧 Aktive Klassifikatoren: ${stats.stats.active_classifiers}`);
        } catch (error) {
            console.warn('Statistiken konnten nicht abgerufen werden:', error);
        }
    }

    private createDemoMessages(): DemoMessage[] {
        return [
            { text: "Leichte Verzögerung bei Türöffnung festgestellt", expected_category: "fahrkabine" },
            { text: "Schmierölstand leicht reduziert – Wartung empfohlen", expected_category: "aufzugsgetriebe" },
            { text: "Vibrationen leicht erhöht – Überprüfung empfohlen", expected_category: "aufzugsgetriebe" },
            { text: "Temperatur im Schaltschrank leicht erhöht", expected_category: "aufzugsgetriebe" },
            { text: "Plötzlicher Systemausfall – Notbetrieb aktiviert", expected_category: "aufzugsgetriebe" },
            { text: "Getriebeölstand nahe Mindestwert", expected_category: "aufzugsgetriebe" },
            { text: "Unregelmäßige Beschleunigung beim Startvorgang", expected_category: "aufzugsgetriebe" },
            { text: "Tür schließt verzögert nach Gewichtserkennung", expected_category: "fahrkabine" },
            { text: "Antrieb benötigt längere Initialisierung", expected_category: "aufzugsgetriebe" },
            { text: "Kein Zugang zur Steuerungseinheit – manuelle Übersteuerung notwendig", expected_category: "aufzugsgetriebe" },
            { text: "Tragseil zeigt Verschleißspuren am Aufhängepunkt", expected_category: "seil" },
            { text: "Seilspannung unregelmäßig - Überprüfung der Seilführung nötig", expected_category: "seil" },
            { text: "Bedienfeld reagiert verzögert auf Tasteneingaben", expected_category: "fahrkabine" },
            { text: "Lichtschranke der Türsicherung defekt", expected_category: "fahrkabine" },
            { text: "Motor läuft unrund - Lagerschaden vermutet", expected_category: "aufzugsgetriebe" }
        ];
    }

    private displayClassificationResult(result: any, expectedCategory?: string): void {
        const classification = result.result;
        const scores = classification.combined_score;
        
        // Finde beste Kategorie
        const bestCategory = Object.keys(scores).reduce((a, b) => 
            scores[a] > scores[b] ? a : b
        );
        const confidence = scores[bestCategory];
        
        console.log(`  🎯 Kategorie: ${bestCategory} (${(confidence * 100).toFixed(1)}%)`);
        
        if (expectedCategory) {
            const correct = bestCategory === expectedCategory;
            console.log(`  ${correct ? '✅' : '❌'} Erwartet: ${expectedCategory}`);
        }
        
        console.log(`  ⏱️  Zeit: ${(classification.processing_time * 1000).toFixed(1)}ms`);
        console.log(`  🔧 Klassifikatoren: ${classification.classifier_count}`);
        
        // Zeige Top-3 Scores
        const topScores = Object.entries(scores as Record<string, number>)
            .sort(([,a], [,b]) => (b as number) - (a as number))
            .slice(0, 3)
            .map(([cat, score]) => `${cat}:${((score as number) * 100).toFixed(0)}%`)
            .join(', ');
        console.log(`  📊 Scores: ${topScores}\n`);
    }

    private async cleanup(): Promise<void> {
        console.log('🛑 Beende Demo System...');
        
        // Disconnecte Client
        if (this.client) {
            this.client.disconnect();
            console.log('🛑 Client getrennt');
        }
        
        // Stoppe Python Server
        if (this.pythonServer) {
            this.pythonServer.kill('SIGTERM');
            
            // Warte kurz auf graceful shutdown
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            if (!this.pythonServer.killed) {
                this.pythonServer.kill('SIGKILL');
                console.log('🔨 Python Server force-killed');
            } else {
                console.log('✅ Python Server gestoppt');
            }
        }
        
        console.log('✅ Demo System beendet');
    }
}

// CLI Interface am Ende der Datei (kann entfernt werden da jetzt in socket_demo.ts)
async function main() {
    const args = process.argv.slice(2);
    const port = args.includes('--port') ? 
        parseInt(args[args.indexOf('--port') + 1]) : 8888;
    
    console.log('🚀 SOCKET-BASIERTES MULTI-CLASSIFIER DEMO');
    console.log('=' .repeat(60));
    
    const demo = new DemoSystem({
        server: { 
            port: port 
        },
        demo: { 
            message_count: 10, 
            delay_between_messages: 500 
        }
    });
    
    try {
        await demo.runDemo();
        process.exit(0);
    } catch (error) {
        console.error('Demo fehlgeschlagen:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}