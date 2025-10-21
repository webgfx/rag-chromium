#!/usr/bin/env python3
"""
One-click RAG integration for GitHub Copilot.
Automatically adds RAG context to your current Copilot chat.

Usage:
    python rag_copilot_auto.py "your question"

This will:
1. Query the RAG system
2. Generate copilot_rag_results.md
3. Display instructions for Copilot to read it
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from copilot_rag_interface import CopilotRAGInterface


def auto_integrate(query: str):
    """Automatically integrate RAG with Copilot."""
    
    print("\n" + "="*70)
    print("🤖 Chromium RAG → GitHub Copilot Auto-Integration")
    print("="*70)
    
    # Step 1: Query RAG
    print(f"\n📊 Step 1/3: Querying RAG for: '{query}'")
    rag = CopilotRAGInterface()
    rag.query(query, top_k=5)
    
    # Step 2: Generate Copilot prompt
    print("\n✍️  Step 2/3: Generating Copilot prompt...")
    
    copilot_prompt = f"""I just searched the Chromium RAG database for "{query}".

The results are saved in copilot_rag_results.md (in this workspace).

Please read that file and provide a detailed explanation based on the real Chromium commits found.

Focus on:
1. Key patterns and implementations
2. Author insights and design decisions  
3. Recent changes and their rationale
4. Code examples from the actual commits
"""
    
    # Save prompt to clipboard if possible
    try:
        import pyperclip
        pyperclip.copy(copilot_prompt)
        print("✅ Copilot prompt copied to clipboard!")
    except:
        print("💡 Copilot prompt (copy this):")
        print("-"*70)
        print(copilot_prompt)
        print("-"*70)
    
    # Step 3: Instructions
    print("\n🎯 Step 3/3: Next Steps")
    print("="*70)
    print("1. Open GitHub Copilot Chat in VS Code")
    print("   (Click the Copilot icon in the sidebar)")
    print()
    print("2. Paste or type:")
    print(f'   "@workspace Read copilot_rag_results.md and explain {query}"')
    print()
    print("3. Copilot will use the RAG results as context!")
    print("="*70)
    
    # Also create a quick-reference file
    with open("COPILOT_PROMPT.txt", "w", encoding="utf-8") as f:
        f.write(copilot_prompt)
    
    print("\n📝 Prompt also saved to: COPILOT_PROMPT.txt")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_copilot_auto.py <your question>")
        print("\nExample:")
        print('  python rag_copilot_auto.py "memory leak fixes in Chrome"')
        print("\nThis will:")
        print("  1. Query the RAG system")
        print("  2. Save results to copilot_rag_results.md")
        print("  3. Generate a ready-to-use Copilot prompt")
        return
    
    query = " ".join(sys.argv[1:])
    auto_integrate(query)


if __name__ == "__main__":
    main()
