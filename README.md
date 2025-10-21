# Chromium RAG System

An advanced Retrieval Augmented Generation (RAG) system for Chromium development, leveraging GPU acceleration and state-of-the-art embedding models to search and understand the Chromium codebase through its commit history.

## 🚀 Features

- **GPU-Accelerated Embeddings**: NVIDIA RTX 5080 (16GB) with CUDA 13.0 for fast embedding generation
- **Advanced Retrieval**: Semantic search with BGE-large-en-v1.5 (1024-dim) embeddings
- **Large-Scale Processing**: Handles 500K+ commits with batch processing and resume capability
- **Real-time Monitoring**: Streamlit dashboard for ingestion progress and system stats
- **GitHub Copilot Integration**: Query interface to enhance Copilot with real Chromium knowledge
- **Production-Ready**: ChromaDB vector database with persistent storage and efficient search

## 📊 Current Status

**Database**: 152,142 documents indexed from Chromium commits (2022-present)
- Phase 1: ✅ 20,000 commits (100%)
- Phase 2: ✅ 40,000 commits (100%) 
- Phase 3: ⏳ 76,000/100,000 commits (76%)

## 🛠️ System Requirements

- **Python**: 3.13+ (tested on 3.13.9)
- **GPU**: NVIDIA GPU with 16GB+ VRAM, CUDA 12.1+
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ for database and model cache
- **Chromium repo**: Local clone of Chromium source

## 📦 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rag-chromium
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment (optional):
```bash
cp .env.example .env
# Edit .env if needed
```

## 🎯 Quick Start

### 1. Query the Existing Database (Recommended)

If you already have data indexed, start querying immediately:

```bash
# Interactive RAG interface
python interactive_rag.py

# Query from command line
python scripts/query_vectors.py --collection chromium_complete --query "memory leak fixes"

# GitHub Copilot integration
python copilot_rag_interface.py "how does Chrome handle WebGL?"
```

### 2. Ingest More Chromium Data

Continue ingesting commits from your Chromium repository:

```bash
# Resume Phase 3 (or start new phase)
python massive_chromium_ingestion.py \
  --repo-path "d:\r\cr\src" \
  --start-date "2022-01-01" \
  --max-commits 100000 \
  --batch-size 1000 \
  --embedding-batch-size 128 \
  --max-workers 8

# Monitor progress with dashboard
streamlit run ingestion_monitor.py
```

### 3. Verify System Status

```bash
# Verify database health and consistency
python verify_database.py

# Rebuild status.json from database (if corrupted)
python rebuild_status.py

# Check database and resume status
python verify_resume.py

# Quick status overview
python quick_resume.py
```

### 4. Database Recovery (Important!)

If `status.json` is corrupted or lost, you can always rebuild it from the database:

```bash
# The database is the source of truth
python rebuild_status.py

# This scans all documents in Qdrant and reconstructs:
# - All processed commit ranges
# - Accurate document counts
# - Complete timeline of what's been processed
```

See [DATABASE_RECOVERY.md](DATABASE_RECOVERY.md) for detailed recovery procedures.

## 📁 Project Structure

```
rag-chromium/
├── rag_system/              # Core RAG system modules
│   ├── core/                # Configuration, logging
│   ├── data/                # Data ingestion, preprocessing, chunking
│   ├── embeddings/          # GPU-accelerated embedding generation
│   ├── retrieval/           # Advanced retrieval with re-ranking
│   ├── vector/              # ChromaDB vector database interface
│   └── generation/          # Response generation (future)
├── scripts/                 # Utility scripts
│   ├── query_vectors.py     # Query interface
│   ├── generate_embeddings.py
│   └── validate_system.py
├── data/                    # Data storage (gitignored)
│   ├── cache/vector_db/     # ChromaDB persistent storage
│   ├── embeddings/          # Cached embeddings
│   └── massive_cache/       # Ingestion progress files
├── logs/                    # Application logs
├── massive_chromium_ingestion.py  # Main ingestion pipeline
├── interactive_rag.py       # Interactive query interface
├── copilot_rag_interface.py # GitHub Copilot integration
├── ingestion_monitor.py     # Streamlit monitoring dashboard
├── verify_database.py       # Database verification tool
├── rebuild_status.py        # Status reconstruction from database
├── verify_resume.py         # Resume verification tool
├── quick_resume.py          # Quick status display
├── README.md                # This file
├── COMPLETE_INGESTION_GUIDE.md  # Detailed ingestion guide
├── COPILOT_INTEGRATION.md   # Copilot integration guide
├── UNIFIED_STATUS_ARCHITECTURE.md  # Status file architecture
├── DATABASE_RECOVERY.md     # Database scanning and recovery guide
├── RESUME_GUIDE.md          # Resume functionality guide
└── requirements.txt         # Python dependencies
```

## 🔧 Key Components

### Ingestion Pipeline (`massive_chromium_ingestion.py`)
- Processes Chromium commits in configurable batches
- GPU-accelerated embedding generation (BGE-large-en-v1.5)
- Automatic resume from interruption
- Multi-phase approach for handling 500K+ commits
- Progress tracking and statistics

### Vector Database (`rag_system/vector/database.py`)
- ChromaDB backend with persistent storage
- Efficient similarity search with pre-computed embeddings
- Metadata filtering and hybrid retrieval
- Collection: `chromium_complete`

### Retrieval System (`rag_system/retrieval/retriever.py`)
- Semantic search using dense embeddings
- Optional re-ranking for improved relevance
- Context-aware query processing
- Multiple retrieval strategies (semantic, hybrid, multi-stage)

### Monitoring Dashboard (`ingestion_monitor.py`)
- Real-time ingestion progress
- Phase completion tracking
- Processing rate and time estimates
- GPU memory and performance metrics

## 📚 Documentation

- **[COMPLETE_INGESTION_GUIDE.md](COMPLETE_INGESTION_GUIDE.md)**: Full guide to ingesting Chromium data
- **[UNIFIED_STATUS_ARCHITECTURE.md](UNIFIED_STATUS_ARCHITECTURE.md)**: Status file architecture and real-time updates
- **[DATABASE_RECOVERY.md](DATABASE_RECOVERY.md)**: Database scanning, verification, and status reconstruction
- **[COPILOT_INTEGRATION.md](COPILOT_INTEGRATION.md)**: How to use with GitHub Copilot
- **[RESUME_GUIDE.md](RESUME_GUIDE.md)**: Resume functionality and troubleshooting

## 🤝 GitHub Copilot Integration with MCP

### 🎯 Native Copilot Integration (Recommended!)

The easiest way to use this RAG system is through the **MCP (Model Context Protocol)** server, which integrates directly into GitHub Copilot.

**Quick Setup (2 minutes):**

1. **Configure VS Code** - Add to your `settings.json`:
```json
{
  "github.copilot.chat.mcp.servers": {
    "chromium-rag": {
      "command": "python",
      "args": ["E:\\rag-chromium\\dist\\chromium-rag-mcp.py"],
      "env": {}
    }
  }
}
```

2. **Restart VS Code**

3. **Start asking questions:**
```
@chromium-rag How does V8 handle garbage collection?
@chromium-rag Explain Chrome's security sandbox
@chromium-rag Show me WebGL rendering implementation
```

📖 **See [`dist/QUICK_START.md`](dist/QUICK_START.md) for complete setup instructions!**

### Alternative Methods

#### Method 1: Direct CLI Query
```bash
# Query and save results to copilot_rag_results.md
python copilot_rag_interface.py "memory management in Chrome"

# Then ask Copilot:
# "@workspace Read copilot_rag_results.md and explain"
```

#### Method 2: Interactive Mode
```bash
python interactive_rag.py
# Ask questions interactively
```

### 📦 Distribution Package

Everything you need for Copilot integration is in the **`/dist`** folder:

```
dist/
├── chromium-rag-mcp.py          # MCP server
├── QUICK_START.md               # 2-minute setup guide
├── README.md                     # Detailed documentation
├── install.bat                   # Windows installer
├── test_mcp.py                   # Verification script
└── vscode-settings-example.json # Settings template
```

**Test the server:**
```bash
cd dist
python test_mcp.py
```

## 🔍 Usage Examples

### Interactive Querying
```bash
python interactive_rag.py
# Ask questions about Chromium development
```

### Command-line Query
```bash
python scripts/query_vectors.py \
  --collection chromium_complete \
  --query "WebGL security improvements" \
  --n-results 10
```

### Continue Ingestion
```bash
# Check current status
python quick_resume.py

# Resume from last checkpoint
python massive_chromium_ingestion.py \
  --repo-path "d:\r\cr\src" \
  --start-date "2022-01-01" \
  --max-commits 100000 \
  --batch-size 1000 \
  --embedding-batch-size 128 \
  --max-workers 8
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Embedding models and batch sizes
- Vector database settings
- Retrieval parameters
- Logging levels

## 🐛 Troubleshooting

**GPU Out of Memory**: Reduce `--embedding-batch-size` (try 64 or 32)

**Slow Ingestion**: Check GPU utilization, adjust `--max-workers`

**Status File Corrupted**: Run `python rebuild_status.py` to reconstruct from database

**Database Mismatch**: Run `python verify_database.py` to check consistency

**Resume Issues**: Run `python verify_resume.py` to check status

**Query Errors**: Ensure collection name matches: `chromium_complete`

**System Crash Recovery**: Run `python rebuild_status.py` to recover state from database

See [DATABASE_RECOVERY.md](DATABASE_RECOVERY.md) for detailed recovery procedures.

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **BGE Models**: BAAI/bge-large-en-v1.5 for embeddings
- **ChromaDB**: Vector database backend
- **PyTorch**: GPU acceleration
- **Chromium Project**: Source data