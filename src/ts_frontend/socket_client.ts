// socket_client.ts - Sauberer Socket Client
import * as net from 'net';
import { EventEmitter } from 'events';

interface ClientConfig {
    host: string;
    port: number;
    timeout: number;
    reconnect: {
        enabled: boolean;
        max_attempts: number;
        delay_ms: number;
        backoff_factor: number;
    };
}

// Hilfsfunktion für Deep Partial
type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

interface SocketMessage {
    type: string;
    request_id?: string;
    [key: string]: any;
}

export class SocketClassifierClient extends EventEmitter {
    private socket: net.Socket | null = null;
    private connected = false;
    private config: ClientConfig;
    private reconnectAttempts = 0;
    private requestId = 0;
    private pendingRequests = new Map<string, {
        resolve: (value: any) => void;
        reject: (error: Error) => void;
        timeout: NodeJS.Timeout;
    }>();

    constructor(config: DeepPartial<ClientConfig> = {}) {
        super();
        
        this.config = {
            host: 'localhost',
            port: 8888,
            timeout: 5000,
            reconnect: {
                enabled: true,
                max_attempts: 3,
                delay_ms: 2000,
                backoff_factor: 1.5,
                ...config.reconnect
            },
            ...config
        } as ClientConfig;
    }

    async connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            console.log(`🔌 Verbinde zu Socket-Server ${this.config.host}:${this.config.port}...`);
            
            this.socket = new net.Socket();
            
            // Connection Events
            this.socket.on('connect', () => {
                this.connected = true;
                this.reconnectAttempts = 0;
                console.log('✅ Socket-Verbindung hergestellt');
                this.emit('connected');
                resolve();
            });

            this.socket.on('data', (data: Buffer) => {
                this.handleIncomingData(data);
            });

            this.socket.on('close', () => {
                this.connected = false;
                console.log('🔌 Socket-Verbindung geschlossen');
                this.emit('disconnected');
                this.handleReconnect();
            });

            this.socket.on('error', (error: Error) => {
                console.error('❌ Socket Fehler:', error);
                this.emit('error', error);
                if (!this.connected) {
                    reject(error);
                }
            });

            // Verbinde
            this.socket.connect(this.config.port, this.config.host);
            
            // Connection Timeout
            setTimeout(() => {
                if (!this.connected) {
                    reject(new Error('Connection timeout'));
                }
            }, this.config.timeout);
        });
    }

    private handleIncomingData(data: Buffer): void {
        try {
            // Message Framing: Lese Länge (4 Bytes)
            if (data.length < 4) {
                console.warn('Datenpaket zu klein für Message Framing');
                return;
            }

            let offset = 0;
            while (offset < data.length) {
                // Lese Nachrichtenlänge
                if (offset + 4 > data.length) break;
                
                const messageLength = data.readUInt32BE(offset);
                offset += 4;

                // Lese Nachricht
                if (offset + messageLength > data.length) break;
                
                const messageData = data.subarray(offset, offset + messageLength);
                offset += messageLength;

                // Parse JSON
                const message = JSON.parse(messageData.toString('utf-8'));
                this.handleMessage(message);
            }
        } catch (error) {
            console.error('Fehler beim Verarbeiten der Daten:', error);
        }
    }

    private handleMessage(message: SocketMessage): void {
        const requestId = message.request_id;
        
        if (requestId && this.pendingRequests.has(requestId)) {
            const request = this.pendingRequests.get(requestId)!;
            clearTimeout(request.timeout);
            this.pendingRequests.delete(requestId);
            
            if (message.type === 'error') {
                request.reject(new Error(message.message || 'Unknown error'));
            } else {
                request.resolve(message);
            }
        } else {
            // Unerwartete Nachricht
            console.warn('Unerwartete Nachricht empfangen:', message);
            this.emit('message', message);
        }
    }

    private handleReconnect(): void {
        if (!this.config.reconnect.enabled || this.reconnectAttempts >= this.config.reconnect.max_attempts) {
            return;
        }

        this.reconnectAttempts++;
        const delay = this.config.reconnect.delay_ms * Math.pow(this.config.reconnect.backoff_factor, this.reconnectAttempts - 1);
        
        console.log(`🔄 Plane Reconnect in ${delay}ms (Versuch ${this.reconnectAttempts})`);
        
        setTimeout(async () => {
            console.log(`🔄 Reconnect Versuch ${this.reconnectAttempts}...`);
            try {
                await this.connect();
                console.log('✅ Reconnect erfolgreich');
            } catch (error) {
                console.error(`❌ Reconnect Versuch ${this.reconnectAttempts} fehlgeschlagen:`, error);
            }
        }, delay);
    }

    private async sendMessage(message: SocketMessage): Promise<any> {
        if (!this.connected || !this.socket) {
            throw new Error('Socket nicht verbunden');
        }

        // Generiere Request ID
        const requestId = `req_${Date.now()}_${++this.requestId}`;
        message.request_id = requestId;

        return new Promise((resolve, reject) => {
            // Timeout für Request
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error(`${message.type} request timeout nach ${this.config.timeout}ms`));
            }, this.config.timeout);

            // Speichere Request
            this.pendingRequests.set(requestId, { resolve, reject, timeout });

            try {
                // Serialisiere Message
                const messageData = Buffer.from(JSON.stringify(message), 'utf-8');
                
                // Message Framing: Sende Länge + Daten
                const lengthBuffer = Buffer.allocUnsafe(4);
                lengthBuffer.writeUInt32BE(messageData.length, 0);
                
                this.socket!.write(Buffer.concat([lengthBuffer, messageData]));
            } catch (error) {
                clearTimeout(timeout);
                this.pendingRequests.delete(requestId);
                reject(error);
            }
        });
    }

    // Public API Methods
    async healthCheck(): Promise<any> {
        return this.sendMessage({ type: 'health_check' });
    }

    async classify(message: string, metadata: any = {}): Promise<any> {
        return this.sendMessage({
            type: 'classify',
            event: { message, metadata }
        });
    }

    async getStats(): Promise<any> {
        return this.sendMessage({ type: 'stats' });
    }

    disconnect(): void {
        this.config.reconnect.enabled = false;
        
        // Lehne alle ausstehenden Requests ab
        for (const [requestId, request] of this.pendingRequests.entries()) {
            clearTimeout(request.timeout);
            request.reject(new Error('Client disconnected'));
        }
        this.pendingRequests.clear();

        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
    }

    isConnected(): boolean {
        return this.connected;
    }
}