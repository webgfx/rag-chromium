# Scripts Directory

This directory contains all utility and operational scripts for the Chromium RAG system, organized by category.

## 📁 Directory Structure

### 🚀 deployment/
Scripts for creating and deploying the RAG system to production environments.

- **`create_deployment_package.py`** - Main deployment package creator
  - Creates full, minimal, or code-only packages
  - Usage: `python create_deployment_package.py [A|B|C]`
  - Options: A=Full (13GB), B=Minimal (3GB), C=Code only (50MB)

- **`create-direct-transfer.bat`** - Windows direct file transfer setup
- **`create-windows-archive.bat`** - Windows archive creation for deployment

### 💾 database/
Scripts for database management, backup, restore, and migration.

- **`backup_database.py`** - Backup Qdrant database with compression
- **`restore_database.py`** - Restore database from backup
- **`migrate_to_qdrant.py`** - Migrate from ChromaDB to Qdrant
- **`verify_database.py`** - Verify database integrity and content
- **`analyze_database_dates.py`** - Analyze commit date distribution
- **`purge_and_test.py`** - Clean database and run tests

### 📥 ingestion/
Scripts for ingesting Chromium commits into the vector database.

- **`ingest_by_index.py`** - Ingest commits by index range
  - Usage: `python ingest_by_index.py --repo-path PATH --start-index N --end-index M`
  - Supports resume and checkpoint recovery

- **`massive_chromium_ingestion.py`** - Full-scale ingestion pipeline
  - Handles millions of commits with batching
  - Built-in error handling and progress tracking

- **`quick_resume.py`** - Resume interrupted ingestion
- **`simulate_ingestion.py`** - Test ingestion without database writes

### 📊 monitoring/
Scripts for monitoring system health, performance, and progress.

- **`ingestion_monitor.py`** - Real-time ingestion progress monitoring
- **`autonomous_monitor.py`** - Autonomous system health monitor
- **`auto_rebuild_monitor.py`** - Automatic status rebuild monitor
- **`backup_scheduler.py`** - Automated backup scheduling
- **`monitor_gpu_performance.ps1`** - PowerShell GPU performance monitor

### 🧪 testing/
Scripts for testing RAG functionality and system components.

- **`test_complete_rag.py`** - End-to-end RAG pipeline test
- **`test_qdrant.py`** - Qdrant database connectivity test
- **`test_commit_extraction.py`** - Test commit data extraction
- **`test_gpu_subprocess_interaction.py`** - GPU subprocess interaction test
- **`test_model_load.py`** - Test embedding model loading
- **`test_monitor_data.py`** - Test monitoring data collection
- **`simple_rag_demo.py`** - Simple RAG query demo

### 🔧 utilities/
General-purpose utility scripts for various tasks.

#### Commit Management
- **`add_commit_indices.py`** - Add indices to commits
- **`fix_commit_indices.py`** - Fix corrupted commit indices
- **`find_commit_index.py`** - Find index of specific commit
- **`find_index_range.py`** - Find index range for date range
- **`lookup_commit.py`** - Look up commit by SHA
- **`lookup_commit_by_index.py`** - Look up commit by index
- **`commit_lookup.py`** - General commit lookup utility

#### Range Management
- **`check_all_ranges.py`** - Check all indexed ranges
- **`check_range_overlaps.py`** - Detect overlapping ranges
- **`merge_ranges.py`** - Merge adjacent ranges
- **`optimize_ranges.py`** - Optimize range storage
- **`reconstruct_ranges.py`** - Reconstruct ranges from database
- **`track_commit_ranges.py`** - Track commit range ingestion

#### Bucket Management
- **`build_buckets_from_database.py`** - Build date buckets from database
- **`create_commit_buckets.py`** - Create commit buckets by date
- **`view_buckets.py`** - View existing bucket structure

#### Query & Analysis
- **`quick_query.py`** - Quick RAG query tool
- **`interactive_rag.py`** - Interactive RAG query interface
- **`analyze_date_coverage.py`** - Analyze date coverage in database

#### System Utilities
- **`warm_up_rag.py`** - Pre-load RAG components
- **`verify_resume.py`** - Verify resume capability
- **`rebuild_status.py`** - Rebuild status.json from database

## 🚀 Quick Start Examples

### Create Deployment Package
```bash
cd scripts/deployment
python create_deployment_package.py A  # Full package
```

### Backup Database
```bash
cd scripts/database
python backup_database.py
```

### Ingest Commits
```bash
cd scripts/ingestion
python ingest_by_index.py --repo-path D:\r\cr\src --start-index 0 --end-index 10000
```

### Monitor Ingestion
```bash
cd scripts/monitoring
python ingestion_monitor.py
```

### Test RAG System
```bash
cd scripts/testing
python test_complete_rag.py
```

### Query Database
```bash
cd scripts/utilities
python quick_query.py "V8 engine optimization"
```

## 📋 Original Scripts Location

These scripts were previously located in:
- Root directory (`e:\rag-chromium\`)
- `tools/` directory
- Old `scripts/` directory (generate_embeddings.py, etc.)

All scripts have been consolidated here for better organization.

## 🔄 Migrated from Root

The following scripts were moved from the root directory:
- All ingestion scripts (massive_chromium_ingestion.py, ingest_by_index.py, etc.)
- All monitoring scripts (ingestion_monitor.py, autonomous_monitor.py, etc.)
- All testing scripts (test_complete_rag.py, test_qdrant.py, etc.)
- All database management scripts (backup_database.py, restore_database.py, etc.)
- All utility scripts (lookup_commit.py, find_commit_index.py, etc.)

## 🔄 Migrated from tools/

The following scripts were moved from `tools/` directory:
- commit_lookup.py
- optimize_ranges.py
- rebuild_status.py

## 📝 Notes

- All scripts maintain their original functionality
- Import paths may need adjustment if running from new location
- Use `python -m scripts.category.script_name` for module imports
- Scripts are organized by primary function (some may fit multiple categories)

## 🆘 Support

For script-specific help, run:
```bash
python script_name.py --help
```

Most scripts include built-in help and usage examples.
