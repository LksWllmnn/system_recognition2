// socket_classifier_client.ts - TypeScript Socket Client
import * as net from 'net';
import { EventEmitter } from 'events';

interface Event {
    id: string;
    timestamp: number;
    channel: 'funk' | 'sms' | 'email' | 'sensor' | 'manual' | 'emergency';
    severity: 'info' | 'warnung' | 'bedrohung';
    rawText: string;
    source?: string;
}

interface ClassificationResult {
    category: string;
    confidence: number;
    classifier_type: string;
}

interface ProcessedEvent {
    event: Event;
    embedding_result: ClassificationResult;
    llm_result: ClassificationResult;
    rule_result: ClassificationResult;
    consensus: string;
    certainty_level: 'certain' | 'uncertain' | 'very_uncertain';
    should_save_for_training: boolean;
}

class SocketClassifierClient extends EventEmitter {
    private socket: net.Socket | null = null;
    private connected: boolean = false;
    private pendingRequests: Map<string, {
        resolve: (result: any) => void;
        reject: (error: Error) => void;
        timeout: NodeJS.Timeout;
    }> = new Map();

    constructor(private host: string = 'localhost', private port: number = 8888) {
        super();
    }

    async connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            console.log(`🔌 Verbinde zu Socket-Server ${this.host}:${this.port}...`);
            
            this.socket = new net.Socket();
            
            this.socket.connect(this.port, this.host, () => {
                this.connected = true;
                console.log('✅ Socket-Verbindung hergestellt');
                resolve();
            });

            this.socket.on('data', (data: Buffer) => {
                this.handleResponse(data.toString());
            });

            this.socket.on('error', (error) => {
                console.error('❌ Socket Fehler:', error);
                this.connected = false;
                reject(error);
            });

            this.socket.on('close', () => {
                console.log('🔌 Socket-Verbindung geschlossen');
                this.connected = false;
                this.emit('disconnected');
            });

            // Timeout für Verbindung
            setTimeout(() => {
                if (!this.connected) {
                    reject(new Error('Socket-Verbindung Timeout'));
                }
            }, 5000);
        });
    }

    private handleResponse(data: string) {
        try {
            const response = JSON.parse(data);
            
            if (response.type === 'classification_result') {
                const requestId = response.request_id;
                const pending = this.pendingRequests.get(requestId);
                
                if (pending) {
                    clearTimeout(pending.timeout);
                    this.pendingRequests.delete(requestId);
                    pending.resolve(response.result);
                }
            } else if (response.type === 'error') {
                // Handle error für letzten Request
                const lastRequest = Array.from(this.pendingRequests.values()).pop();
                if (lastRequest) {
                    const requestId = Array.from(this.pendingRequests.keys()).pop()!;
                    clearTimeout(lastRequest.timeout);
                    this.pendingRequests.delete(requestId);
                    lastRequest.reject(new Error(response.message));
                }
            }
        } catch (error) {
            console.error('Fehler beim Parsen der Antwort:', error);
        }
    }

    async classifyEvent(event: Event, timeoutMs: number = 10000): Promise<ProcessedEvent> {
        if (!this.connected || !this.socket) {
            throw new Error('Socket nicht verbunden');
        }

        return new Promise((resolve, reject) => {
            const requestId = `req_${Date.now()}_${Math.random()}`;
            
            // Timeout für Request
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error(`Klassifikation timeout nach ${timeoutMs}ms`));
            }, timeoutMs);

            // Request speichern
            this.pendingRequests.set(requestId, {
                resolve,
                reject,
                timeout
            });

            // Request senden
            const request = {
                type: 'classify',
                request_id: requestId,
                event: event
            };

            this.socket!.write(JSON.stringify(request));
        });
    }

    async healthCheck(): Promise<boolean> {
        if (!this.connected || !this.socket) {
            return false;
        }

        try {
            const request = {
                type: 'health_check'
            };

            this.socket.write(JSON.stringify(request));
            return true;
        } catch (error) {
            return false;
        }
    }

    async getStats(): Promise<{ total: number; by_category: Record<string, number> }> {
        if (!this.connected || !this.socket) {
            throw new Error('Socket nicht verbunden');
        }

        return new Promise((resolve, reject) => {
            const requestId = `stats_${Date.now()}`;
            
            const timeout = setTimeout(() => {
                reject(new Error('Stats request timeout'));
            }, 5000);

            this.pendingRequests.set(requestId, {
                resolve,
                reject,
                timeout
            });

            const request = {
                type: 'stats',
                request_id: requestId
            };

            this.socket!.write(JSON.stringify(request));
        });
    }

    disconnect(): void {
        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
    }
}

class SocketEnhancedSystemRecognizer {
    private classifierClient: SocketClassifierClient;
    private systemState: string = "Default";
    private focusMode: string | null = null;
    private eventCounts: Record<string, number> = {};

    constructor() {
        this.classifierClient = new SocketClassifierClient();
    }

    async initialize(): Promise<void> {
        try {
            await this.classifierClient.connect();
            
            // Teste Verbindung
            const healthy = await this.classifierClient.healthCheck();
            if (!healthy) {
                throw new Error('Health Check fehlgeschlagen');
            }
            
            console.log('🎯 Socket Enhanced System Recognizer bereit');
        } catch (error) {
            console.error('❌ Socket Initialisierung fehlgeschlagen:', error);
            throw error;
        }
    }

    async processEvent(event: Event): Promise<void> {
        console.log(`\n📨 Event erhalten: [${event.severity.toUpperCase()}] - ${event.rawText}`);
        
        try {
            const classificationStart = Date.now();
            const result = await this.classifierClient.classifyEvent(event);
            const classificationTime = Date.now() - classificationStart;

            this.logClassificationResults(result, classificationTime);
            this.updateSystemState(event, result);
            this.logDataQuality(result);

        } catch (error) {
            console.error('❌ Fehler bei der Socket-Klassifikation:', error);
            this.handleClassificationError(event);
        }
    }

    private logClassificationResults(result: ProcessedEvent, processingTime: number): void {
        console.log(`🤖 Socket Multi-Klassifikation (${processingTime}ms):`);
        console.log(`  🔗 Embedding: ${result.embedding_result.category} (${result.embedding_result.confidence})`);
        console.log(`  🧠 LLM: ${result.llm_result.category} (${result.llm_result.confidence})`);
        console.log(`  📏 Regeln: ${result.rule_result.category} (${result.rule_result.confidence})`);
        console.log(`  🎯 Konsensus: ${result.consensus} (${result.certainty_level})`);
        
        if (result.should_save_for_training) {
            console.log(`  💾 Für BERT-Training gespeichert`);
        }
    }

    private updateSystemState(event: Event, result: ProcessedEvent): void {
        const category = result.consensus;

        if (result.certainty_level === 'certain' || event.severity === 'bedrohung') {
            if (event.severity === 'bedrohung' && category !== 'kein_fokus') {
                this.focusMode = category;
                this.systemState = `Notfallmodus aktiv: ${this.focusMode}`;
                console.log(`⚠️ BEDROHUNG ERKANNT! Umschalten auf ${this.focusMode}`);
            } else if (event.severity === 'warnung' && category !== 'kein_fokus') {
                this.eventCounts[category] = (this.eventCounts[category] || 0) + 1;
                
                if (this.eventCounts[category] >= 3) {
                    this.focusMode = category;
                    this.systemState = `Fokusmodus aktiv: ${this.focusMode}`;
                    console.log(`🔍 Mehrere Meldungen zu ${category}. Umschalten in Fokusmodus.`);
                } else {
                    this.systemState = "Default";
                }
            }
        } else {
            console.log(`⚠️ Unsichere Klassifikation (${result.certainty_level}) - keine Systemänderung`);
        }

        console.log(`📌 Systemzustand: ${this.systemState}`);
    }

    private logDataQuality(result: ProcessedEvent): void {
        switch (result.certainty_level) {
            case 'certain':
                console.log(`✅ Hohe Datenqualität - alle Klassifikatoren stimmen überein`);
                break;
            case 'uncertain':
                console.log(`⚠️ Mittlere Datenqualität - ein Klassifikator weicht ab`);
                break;
            case 'very_uncertain':
                console.log(`❌ Niedrige Datenqualität - alle Klassifikatoren unterschiedlich`);
                break;
        }
    }

    private handleClassificationError(event: Event): void {
        console.log(`🚨 Fallback-Verarbeitung für Event: ${event.id}`);
        
        const text = event.rawText.toLowerCase();
        let category = 'kein_fokus';
        
        if (text.includes('kabine') || text.includes('tür') || text.includes('fahrgast')) {
            category = 'fahrstuhl_kabine';
        } else if (text.includes('seil') || text.includes('spannung') || text.includes('umlenkrolle')) {
            category = 'fahrstuhl_seil';
        } else if (text.includes('getriebe') || text.includes('motor') || text.includes('antrieb')) {
            category = 'fahrstuhl_getriebe';
        }
        
        console.log(`📏 Fallback-Klassifikation: ${category}`);
        
        if (event.severity === 'bedrohung' && category !== 'kein_fokus') {
            this.focusMode = category;
            this.systemState = `Notfallmodus aktiv: ${this.focusMode}`;
            console.log(`⚠️ BEDROHUNG ERKANNT! Fallback-Umschaltung auf ${this.focusMode}`);
        }
        
        console.log(`📌 Systemzustand: ${this.systemState}`);
    }

    async getSystemStatus(): Promise<{
        state: string;
        focusMode: string | null;
        eventCounts: Record<string, number>;
        trainingStats: { total: number; by_category: Record<string, number> };
    }> {
        try {
            const trainingStats = await this.classifierClient.getStats();
            
            return {
                state: this.systemState,
                focusMode: this.focusMode,
                eventCounts: { ...this.eventCounts },
                trainingStats
            };
        } catch (error) {
            console.error('Fehler beim Abrufen der Statistiken:', error);
            return {
                state: this.systemState,
                focusMode: this.focusMode,
                eventCounts: { ...this.eventCounts },
                trainingStats: { total: 0, by_category: {} }
            };
        }
    }

    destroy(): void {
        this.classifierClient.disconnect();
        console.log('🛑 Socket Enhanced System Recognizer beendet');
    }
}

export { SocketClassifierClient, SocketEnhancedSystemRecognizer, Event, ProcessedEvent, ClassificationResult };