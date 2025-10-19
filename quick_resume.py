#!/usr/bin/env python3
"""
Quick reference for resuming ingestion after stopping.
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║              CHROMIUM RAG INGESTION - RESUME GUIDE                 ║
╔════════════════════════════════════════════════════════════════════╗

📊 CURRENT STATE:
   • Total in Database: 94,073 documents (accumulated)
   • Last Session: 10,000 commits processed
   • Ready to Resume: YES ✅

🔄 TO RESUME PHASE 3:
   Run the same command you used before:

   python massive_chromium_ingestion.py \\
       --repo-path "d:\\r\\cr\\src" \\
       --start-date "2022-01-01" \\
       --max-commits 100000 \\
       --batch-size 1000 \\
       --embedding-batch-size 128 \\
       --max-workers 8

📋 WHAT HAPPENS:
   1. Loads progress.json (12 batches completed)
   2. Skips batches 1-12 (already done)
   3. Continues from batch 13
   4. Adds to existing 94,073 documents

🛡️ SAFETY:
   • Progress saved after every batch
   • Safe to Ctrl+C at any time
   • No data loss
   • Database is consistent

📈 MONITOR PROGRESS:
   streamlit run ingestion_monitor.py

🔍 VERIFY SYSTEM:
   python verify_resume.py

═══════════════════════════════════════════════════════════════════

KEY FILES:
   • data/massive_cache/progress.json  → Session state
   • data/cache/vector_db/            → Database
   • logs/                            → Error logs

═══════════════════════════════════════════════════════════════════
""")
