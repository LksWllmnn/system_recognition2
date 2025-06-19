// socket_demo.ts - Haupt-Demo Entry Point
import { DemoSystem } from './demo_system';

async function main() {
    const args = process.argv.slice(2);
    
    // Parse Command Line Arguments
    const config: any = {
        server: {},
        client: {},
        demo: {}
    };
    
    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '--port':
                config.server.port = parseInt(args[++i]);
                break;
            case '--host':
                config.server.host = args[++i];
                break;
            case '--timeout':
                config.client.timeout = parseInt(args[++i]);
                break;
            case '--messages':
                config.demo.message_count = parseInt(args[++i]);
                break;
            case '--delay':
                config.demo.delay_between_messages = parseInt(args[++i]);
                break;
            case '--help':
                showHelp();
                process.exit(0);
        }
    }
    
    // Zeige Banner
    console.log('🚀 AUFZUGS-SYSTEM MULTI-CLASSIFIER DEMO');
    console.log('=' .repeat(70));
    console.log('🔧 Aufzugs-Komponenten Klassifikation:');
    console.log('   • Fahrkabine: Türen, Bedienelemente, Innenraum, Sensoren');
    console.log('   • Seil: Tragseile, Führungsseile, Seilrollen, Aufhängung');
    console.log('   • Aufzugsgetriebe: Motor, Getriebe, Schmierung, Steuerung');
    console.log('   • Socket-basierte Kommunikation mit Message Framing');
    console.log('   • Parallele Multi-Klassifikator Verarbeitung');
    console.log('   • LLM-Integration für komplexe Diagnosen');
    console.log('=' .repeat(70));
    
    // Zeige Konfiguration
    const finalConfig = {
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
            delay_between_messages: 800,
            ...config.demo
        }
    };
    
    console.log('📋 Konfiguration:');
    console.log(`   🌐 Server: ${finalConfig.server.host}:${finalConfig.server.port}`);
    console.log(`   ⏱️  Client Timeout: ${finalConfig.client.timeout}ms`);
    console.log(`   📝 Nachrichten: ${finalConfig.demo.message_count}`);
    console.log(`   🕐 Verzögerung: ${finalConfig.demo.delay_between_messages}ms`);
    console.log('');
    
    // Erstelle und starte Demo
    const demo = new DemoSystem(finalConfig);
    
    try {
        const startTime = Date.now();
        await demo.runDemo();
        const duration = (Date.now() - startTime) / 1000;
        
        console.log('');
        console.log('🎉 DEMO ERFOLGREICH ABGESCHLOSSEN!');
        console.log(`⏱️  Gesamtlaufzeit: ${duration.toFixed(1)}s`);
        console.log('');
        console.log('💡 Nächste Schritte:');
        console.log('   • Teste einzelne Module separat');
        console.log('   • Erweitere Klassifikatoren nach Bedarf');
        console.log('   • Integriere in deine Anwendung');
        console.log('   • Siehe README.md für Details');
        
        process.exit(0);
        
    } catch (error) {
        console.error('');
        console.error('❌ DEMO FEHLGESCHLAGEN:');
        console.error('', error);
        console.error('');
        console.error('🔧 TROUBLESHOOTING:');
        console.error('1. Prüfe ob Python installiert ist: python --version');
        console.error('2. Installiere Dependencies: pip install requests');
        console.error('3. Prüfe ob Port frei ist: netstat -an | findstr "8888"');
        console.error('4. Verwende anderen Port: --port 9999');
        console.error('5. Aktiviere Debug-Logs für Details');
        console.error('');
        console.error('📖 Siehe README.md für ausführliche Hilfe');
        
        process.exit(1);
    }
}

function showHelp() {
    console.log('🚀 Socket Multi-Classifier Demo');
    console.log('');
    console.log('Usage: npx ts-node src/socket_demo.ts [options]');
    console.log('');
    console.log('Optionen:');
    console.log('  --port <number>     Server Port (default: 8888)');
    console.log('  --host <string>     Server Host (default: localhost)');
    console.log('  --timeout <number>  Client Timeout in ms (default: 5000)');
    console.log('  --messages <number> Anzahl Demo-Nachrichten (default: 10)');
    console.log('  --delay <number>    Verzögerung zwischen Nachrichten in ms (default: 800)');
    console.log('  --help              Diese Hilfe anzeigen');
    console.log('');
    console.log('Beispiele:');
    console.log('  npx ts-node src/socket_demo.ts');
    console.log('  npx ts-node src/socket_demo.ts --port 9999 --messages 5');
    console.log('  npx ts-node src/socket_demo.ts --timeout 10000 --delay 500');
    console.log('');
    console.log('Debug-Mode:');
    console.log('  DEBUG=* npx ts-node src/socket_demo.ts');
}

// Graceful Shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Demo wurde durch Benutzer abgebrochen');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Demo wurde beendet');
    process.exit(0);
});

// Unhandled Errors
process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection:', reason);
    process.exit(1);
});

process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
});

// Starte Demo
if (require.main === module) {
    main();
}