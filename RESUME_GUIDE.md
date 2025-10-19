# Resume Functionality Guide

## How Resume Works

The ingestion pipeline **automatically saves and resumes progress**. You can safely stop and restart the ingestion at any time.

## What Gets Saved

The pipeline saves progress after every batch to `data/massive_cache/progress.json`:

```json
{
  "commits_processed": 10000,
  "batches_completed": 12,
  "documents_created": 10007,
  "embeddings_generated": 10007,
  "last_batch_time": 18.09,
  "session_start": 1760878676.96,
  "first_commit_sha": "abc123...",
  "last_commit_sha": "def456...",
  "start_date": "2022-01-01",
  "end_date": null,
  "max_commits": 100000
}
```

## How to Stop Ingestion

You can stop the ingestion at any time using:
- `Ctrl+C` in the terminal
- Closing the terminal
- Killing the process

The last completed batch will be saved, and no data will be lost.

## How to Resume

Simply run the **exact same command** you used before:

```bash
python massive_chromium_ingestion.py --repo-path "d:\r\cr\src" --start-date "2022-01-01" --max-commits 100000 --batch-size 1000 --embedding-batch-size 128 --max-workers 8
```

### What Happens on Resume

1. ✅ Loads `progress.json` to see where you stopped
2. ✅ Checks database for actual document count
3. ✅ Skips already-processed batches (line 421-423)
4. ✅ Continues from next batch
5. ✅ Preserves all commit range information

### Resume Logic

```python
# From massive_chromium_ingestion.py line 421-423
if batch_count <= progress.get('batches_completed', 0):
    self.logger.info(f"Skipping already processed batch {batch_count}")
    continue
```

## Important Notes

### ✅ Safe to Resume
- Stop at any time
- Same command parameters must be used
- Progress file tracks everything
- Database stores all documents safely

### ⚠️ Don't Do This
- **Don't change parameters** between runs (start_date, max_commits, etc.)
- **Don't delete progress.json** unless starting fresh
- **Don't change batch_size** between runs (may cause batch mismatch)

## Starting Fresh

To start a completely new ingestion run:

```bash
# Clear progress
Remove-Item -Path "e:\rag-chromium\data\massive_cache\progress.json" -ErrorAction SilentlyContinue

# Run ingestion with new parameters
python massive_chromium_ingestion.py --repo-path "d:\r\cr\src" --start-date "2022-01-01" --max-commits 100000 --batch-size 1000 --embedding-batch-size 128 --max-workers 8
```

## Verifying Resume Works

### Check Current Progress
```bash
python -c "import json; data = json.load(open('data/massive_cache/progress.json')); print(f'Batches: {data[\"batches_completed\"]}, Commits: {data[\"commits_processed\"]}')"
```

### Check Database
```bash
python check_db_size.py
```

### Compare Numbers
- **Progress file**: Shows session progress (10,000 commits)
- **Database**: Shows total accumulated (94,073 documents)
- **Resume**: Will add to the 94,073 total

## Example Resume Flow

### First Run
```bash
# Start ingestion
python massive_chromium_ingestion.py ... --max-commits 100000

# Processes 20,000 commits → stopped with Ctrl+C
# progress.json shows: batches_completed: 20
```

### Resume
```bash
# Same command
python massive_chromium_ingestion.py ... --max-commits 100000

# Output shows:
# "Resuming from: 20000 commits processed"
# "Skipping already processed batch 1"
# "Skipping already processed batch 2"
# ...
# "Skipping already processed batch 20"
# "Processing batch 21 (1000 commits)"  ← Continues here!
```

## Monitoring During Resume

Use the Streamlit dashboard to watch progress:

```bash
streamlit run ingestion_monitor.py
```

The dashboard shows:
- **Overall**: 94,073 total accumulated documents
- **Current Session**: New commits being added
- **Phase Progress**: Updated in real-time

## Recovery from Errors

If the process crashes or errors occur:

1. Check `data/massive_cache/errors.log`
2. Fix the issue (e.g., disk space, memory)
3. Resume with same command
4. Pipeline will skip processed batches and continue

## Summary

✅ **You can safely stop and resume at any time**  
✅ **No data loss - every batch is saved**  
✅ **Use the same command to resume**  
✅ **Progress is tracked automatically**  
✅ **Dashboard shows both session and overall progress**
