// GitHub Copilot Chat Participant for Chromium RAG
// Place this in .vscode or use as reference for custom chat participant

const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

/**
 * Chromium RAG Chat Participant
 * Automatically queries the RAG system when Chromium-related questions are asked
 */
class ChromiumRAGParticipant {
    constructor() {
        this.id = 'chromium-rag';
        this.name = 'Chromium RAG';
        this.description = 'Query Chromium codebase using RAG (150K+ commits)';
        this.resultsFile = path.join(vscode.workspace.rootPath || '', 'copilot_rag_results.md');
    }

    /**
     * Check if query is Chromium-related
     */
    isChromiumQuery(query) {
        const keywords = [
            'chromium', 'chrome', 'blink', 'v8', 'webkit',
            'browser', 'renderer', 'webgl', 'gpu', 'skia',
            'compositor', 'ipc', 'mojo', 'content', 'views'
        ];
        
        const lowerQuery = query.toLowerCase();
        return keywords.some(keyword => lowerQuery.includes(keyword));
    }

    /**
     * Execute RAG query
     */
    async queryRAG(query, topK = 5) {
        return new Promise((resolve, reject) => {
            const pythonScript = path.join(vscode.workspace.rootPath || '', 'copilot_rag_interface.py');
            const command = `python "${pythonScript}" "${query}"`;

            exec(command, { cwd: vscode.workspace.rootPath }, (error, stdout, stderr) => {
                if (error) {
                    reject(error);
                    return;
                }
                
                // Read results file
                if (fs.existsSync(this.resultsFile)) {
                    const results = fs.readFileSync(this.resultsFile, 'utf-8');
                    resolve(results);
                } else {
                    resolve('No results found.');
                }
            });
        });
    }

    /**
     * Handle chat request
     */
    async handleRequest(request, context, stream, token) {
        const query = request.prompt;

        // Show progress
        stream.progress('Searching Chromium RAG database (150K+ commits)...');

        try {
            // Query RAG
            const results = await this.queryRAG(query);

            // Stream results
            stream.markdown(`## Chromium RAG Results\n\n${results}`);
            
            // Add reference
            stream.reference(vscode.Uri.file(this.resultsFile));
            
            return { metadata: { command: 'chromium-rag-query' } };
            
        } catch (error) {
            stream.markdown(`❌ Error querying RAG: ${error.message}`);
            return { metadata: { command: 'error' } };
        }
    }
}

/**
 * Activate extension
 */
function activate(context) {
    const participant = new ChromiumRAGParticipant();
    
    // Register chat participant
    const chatParticipant = vscode.chat.createChatParticipant(
        participant.id,
        async (request, context, stream, token) => {
            return await participant.handleRequest(request, context, stream, token);
        }
    );

    chatParticipant.iconPath = vscode.Uri.file(
        path.join(context.extensionPath, 'icon.png')
    );

    context.subscriptions.push(chatParticipant);

    // Register commands
    const queryCommand = vscode.commands.registerCommand(
        'chromium-rag.query',
        async () => {
            const query = await vscode.window.showInputBox({
                prompt: 'Enter Chromium query',
                placeHolder: 'e.g., How does Chrome handle memory leaks?'
            });

            if (query) {
                await participant.queryRAG(query);
                
                // Open results
                const doc = await vscode.workspace.openTextDocument(participant.resultsFile);
                await vscode.window.showTextDocument(doc, { preview: false });
            }
        }
    );

    context.subscriptions.push(queryCommand);
}

module.exports = { activate };
