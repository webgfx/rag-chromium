#!/usr/bin/env python3
"""
Automatic RAG integration via GitHub Copilot prompt enhancement.
This script monitors Copilot's context and automatically injects RAG results.

Usage:
    1. Run in background: python auto_rag_prompt.py
    2. Ask Copilot questions naturally
    3. Script detects Chromium-related queries and auto-queries RAG
"""

import sys
import re
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from copilot_rag_interface import CopilotRAGInterface


class CopilotContextMonitor(FileSystemEventHandler):
    """Monitor for Copilot context and auto-query RAG."""
    
    def __init__(self):
        self.rag = None
        self.chromium_keywords = [
            'chromium', 'chrome', 'blink', 'v8', 'webkit', 'skia',
            'webgl', 'webrtc', 'ipc', 'mojo', 'renderer', 'browser',
            'gpu', 'compositor', 'devtools', 'service worker'
        ]
        self.last_query = None
        self.last_query_time = 0
    
    def _get_rag(self):
        """Lazy load RAG."""
        if self.rag is None:
            print("🚀 Initializing RAG system...")
            self.rag = CopilotRAGInterface()
        return self.rag
    
    def is_chromium_related(self, text):
        """Check if text is Chromium-related."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.chromium_keywords)
    
    def extract_query(self, text):
        """Extract query from Copilot context."""
        # Look for question patterns
        patterns = [
            r'how (?:does|do|is|are) .*\?',
            r'what (?:is|are|does|do) .*\?',
            r'why (?:does|do|is|are) .*\?',
            r'explain .*',
            r'describe .*',
            r'tell me about .*'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip('?')
        
        return None
    
    def auto_query_rag(self, query):
        """Automatically query RAG if relevant."""
        # Avoid duplicate queries
        current_time = time.time()
        if query == self.last_query and (current_time - self.last_query_time) < 60:
            return
        
        self.last_query = query
        self.last_query_time = current_time
        
        print(f"\n🔍 Auto-detected Chromium query: {query}")
        print("📡 Querying RAG system...")
        
        rag = self._get_rag()
        rag.query(query, top_k=5)
        
        print("✅ RAG results ready in copilot_rag_results.md")
        print("💡 Copilot can now use these results for context!\n")
    
    def on_modified(self, event):
        """Monitor file modifications for Copilot context."""
        if event.is_directory:
            return
        
        # Monitor specific files that might contain Copilot queries
        if event.src_path.endswith(('.py', '.js', '.ts', '.cpp', '.cc', '.h')):
            try:
                with open(event.src_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for Chromium-related content
                if self.is_chromium_related(content):
                    query = self.extract_query(content)
                    if query and len(query) > 10:  # Minimum query length
                        self.auto_query_rag(query)
            
            except Exception as e:
                pass  # Silently ignore file read errors


def monitor_workspace():
    """Monitor workspace for Copilot activity."""
    print("🤖 Chromium RAG Auto-Integration")
    print("="*60)
    print("Monitoring workspace for Chromium-related questions...")
    print("When Copilot context includes Chromium queries, RAG is auto-queried.")
    print("\nPress Ctrl+C to stop.\n")
    
    event_handler = CopilotContextMonitor()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n\n✅ Auto-integration stopped.")
    
    observer.join()


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print(__doc__)
        return
    
    # Check if watchdog is installed
    try:
        import watchdog
    except ImportError:
        print("❌ Error: 'watchdog' package required")
        print("Install with: pip install watchdog")
        return
    
    monitor_workspace()


if __name__ == "__main__":
    main()
