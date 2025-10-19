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
# Check database and resume status
python verify_resume.py

# Quick status overview
python quick_resume.py
```

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
├── verify_resume.py         # Resume verification tool
├── quick_resume.py          # Quick status display
├── README.md                # This file
├── COMPLETE_INGESTION_GUIDE.md  # Detailed ingestion guide
├── COPILOT_INTEGRATION.md   # Copilot integration guide
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
- **[COPILOT_INTEGRATION.md](COPILOT_INTEGRATION.md)**: How to use with GitHub Copilot
- **[RESUME_GUIDE.md](RESUME_GUIDE.md)**: Resume functionality and troubleshooting

## 🤝 GitHub Copilot Integration (Automatic!)

### Quick Start - 3 Ways to Use:

#### Method 1: One-Command Auto-Integration (Easiest!)
```bash
# Automatically query RAG and get Copilot prompt ready
python rag_copilot_auto.py "memory management in Chrome"

# Then paste the generated prompt into Copilot Chat
# (It's copied to your clipboard automatically!)
```

#### Method 2: VS Code Task (Keyboard Shortcut)
1. Press `Ctrl+Shift+R` (or use Tasks menu)
2. Enter your question
3. Results open automatically
4. Ask Copilot: `@workspace Read copilot_rag_results.md and explain`

#### Method 3: Direct Query
```bash
# Query and save results
python copilot_rag_interface.py "memory management in Chrome"

# Then in Copilot Chat:
# "@workspace Read copilot_rag_results.md and explain the memory management patterns"
```

### 🎯 Automatic Context Loading

Copilot is configured to automatically use RAG results! Just:
1. Run a query (any method above)
2. Ask Copilot about your topic
3. Copilot will use `copilot_rag_results.md` as context

See [COPILOT_INTEGRATION.md](COPILOT_INTEGRATION.md) for detailed workflows.

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

**Resume Issues**: Run `python verify_resume.py` to check status

**Query Errors**: Ensure collection name matches: `chromium_complete`

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- **BGE Models**: BAAI/bge-large-en-v1.5 for embeddings
- **ChromaDB**: Vector database backend
- **PyTorch**: GPU acceleration
- **Chromium Project**: Source data