"""
Find the actual commit index range in our database.
"""
from qdrant_client import QdrantClient
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def find_index_range():
    print("Scanning database for commit index range...\n")
    
    client = QdrantClient(path='data/cache/qdrant_db')
    
    # Scroll through all records
    offset = None
    all_indices = []
    batch_count = 0
    
    while True:
        records, next_offset = client.scroll(
            'chromium_complete',
            limit=1000,
            offset=offset,
            with_payload=['commit_index'],
            with_vectors=False
        )
        
        if not records:
            break
            
        batch_count += 1
        for record in records:
            if 'commit_index' in record.payload:
                all_indices.append(record.payload['commit_index'])
        
        print(f"Processed batch {batch_count}: {len(all_indices)} indices so far...")
        
        offset = next_offset
        if offset is None:
            break
    
    client.close()
    
    if not all_indices:
        print("❌ No commit indices found!")
        return
    
    unique_indices = sorted(set(all_indices))
    
    print(f"\n{'='*60}")
    print(f"DATABASE COMMIT INDEX RANGE")
    print(f"{'='*60}")
    print(f"Total documents: {len(all_indices)}")
    print(f"Unique commits: {len(unique_indices)}")
    print(f"Min index: {min(unique_indices)}")
    print(f"Max index: {max(unique_indices)}")
    print(f"\nChromium total: 1,755,640 commits")
    print(f"Our coverage: {len(unique_indices)/1755640*100:.2f}%")
    print(f"\n✅ Valid query range: {min(unique_indices)} to {max(unique_indices)}")
    
    # Show some sample valid indices
    print(f"\nSample valid indices to test:")
    samples = [
        unique_indices[0],
        unique_indices[len(unique_indices)//4],
        unique_indices[len(unique_indices)//2],
        unique_indices[3*len(unique_indices)//4],
        unique_indices[-1]
    ]
    for idx in samples:
        print(f"  {idx}")

if __name__ == '__main__':
    find_index_range()
