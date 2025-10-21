# Quick Scripts Reference

All scripts have been organized into the `scripts/` directory. Here are the most commonly used commands:

## 🚀 Create Deployment Package

```bash
# Full package (database + models + code) - 13.47 GB
python scripts/deployment/create_deployment_package.py A

# Minimal package (database + code) - ~3 GB
python scripts/deployment/create_deployment_package.py B

# Code only package - ~50 MB
python scripts/deployment/create_deployment_package.py C
```

**Output:** `deployment/chromium-rag-full-YYYYMMDD/` with complete deployment package

## 💾 Database Management

```bash
# Backup database
python scripts/database/backup_database.py

# Restore from backup
python scripts/database/restore_database.py

# Verify database integrity
python scripts/database/verify_database.py

# Analyze date coverage
python scripts/database/analyze_database_dates.py
```

## 📥 Ingestion

```bash
# Ingest by index range
python scripts/ingestion/ingest_by_index.py --repo-path D:\r\cr\src --start-index 0 --end-index 10000

# Full ingestion pipeline
python scripts/ingestion/massive_chromium_ingestion.py

# Resume interrupted ingestion
python scripts/ingestion/quick_resume.py
```

## 📊 Monitoring

```bash
# Monitor ingestion progress
python scripts/monitoring/ingestion_monitor.py

# Autonomous system monitor
python scripts/monitoring/autonomous_monitor.py

# Schedule backups
python scripts/monitoring/backup_scheduler.py
```

## 🧪 Testing

```bash
# End-to-end RAG test
python scripts/testing/test_complete_rag.py

# Test database connection
python scripts/testing/test_qdrant.py

# Simple RAG demo
python scripts/testing/simple_rag_demo.py
```

## 🔧 Utilities

```bash
# Quick query
python scripts/utilities/quick_query.py "V8 engine optimization"

# Interactive RAG
python scripts/utilities/interactive_rag.py

# Lookup commit by SHA
python scripts/utilities/lookup_commit.py abc123def456

# Lookup commit by index
python scripts/utilities/lookup_commit_by_index.py 1234567

# Find commit index
python scripts/utilities/find_commit_index.py abc123def456

# Rebuild status
python scripts/utilities/rebuild_status.py

# View commit buckets
python scripts/utilities/view_buckets.py
```

## 📁 Directory Structure

```
scripts/
├── deployment/      # Deployment package creation (3 files)
├── database/        # Database management (6 files)
├── ingestion/       # Commit ingestion (4 files)
├── monitoring/      # System monitoring (5 files)
├── testing/         # RAG testing (7 files)
└── utilities/       # General utilities (22 files)
```

## 📖 Full Documentation

See `scripts/README.md` for complete documentation of all 56 scripts.

## 🔗 Related Files

- `copilot_rag_interface.py` - RAG query interface (root directory)
- `config.yaml` - System configuration
- `requirements.txt` - All dependencies (Python 3.13 compatible, some packages commented out)

## 📦 Deployment Package Contents

When you create a deployment package, it includes:

- ✅ Vector database (2.47 GB, 244,403 commits)
- ✅ Embedding models (11 GB, BAAI/bge-large-en-v1.5)
- ✅ Server & client code (`dist/` folder)
- ✅ RAG system modules (`rag_system/` folder)
- ✅ All utility scripts (`scripts/` folder)
- ✅ Windows deployment scripts (quick-deploy.bat, etc.)
- ✅ Complete documentation (README, setup guides)
- ✅ Configuration files (config.yaml, .env.example)

## 🆘 Getting Help

Most scripts support `--help`:

```bash
python scripts/ingestion/ingest_by_index.py --help
```

For detailed information about each script category:

```bash
# View all scripts in a category
ls scripts/deployment/
ls scripts/database/
ls scripts/ingestion/
ls scripts/monitoring/
ls scripts/testing/
ls scripts/utilities/
```
