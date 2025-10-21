#!/usr/bin/env python3
"""
Simulate the exact ingestion scenario to reproduce the KeyboardInterrupt.
This mimics what happens during actual ingestion.
"""

import subprocess
import time
import sys

def simulate_ingestion_batch():
    """Simulate what happens during ingestion of a batch."""
    repo_path = r"d:\r\cr\src"
    
    print("Simulating ingestion batch processing...")
    print("="*80)
    
    # Step 1: Get commits like the cache does
    print("\nStep 1: Fetching commit list (like cache)...")
    try:
        start = time.time()
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1755000', '--max-count', '500'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        elapsed = time.time() - start
        batch_shas = result.stdout.strip().split('\n')
        print(f"✓ Fetched {len(batch_shas)} commits in {elapsed:.2f}s")
    except KeyboardInterrupt as e:
        print(f"✗ INTERRUPTED at Step 1: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR at Step 1: {e}")
        return False
    
    time.sleep(0.5)
    
    # Step 2: Fetch commit details (like the pipeline does)
    print(f"\nStep 2: Fetching details for {len(batch_shas)} commits...")
    try:
        start = time.time()
        format_str = '%H%n%an%n%ae%n%ai%n%B%n--END--'
        result = subprocess.run(
            ['git', 'log', '--format=' + format_str, '--no-walk'] + batch_shas,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        elapsed = time.time() - start
        print(f"✓ Fetched commit details in {elapsed:.2f}s")
        print(f"  Output size: {len(result.stdout):,} bytes")
    except KeyboardInterrupt as e:
        print(f"✗ INTERRUPTED at Step 2 after {time.time() - start:.2f}s: {e}")
        print(f"  This is where the issue occurs!")
        return False
    except Exception as e:
        print(f"✗ ERROR at Step 2: {e}")
        return False
    
    time.sleep(0.5)
    
    # Step 3: Get stats for first few commits (like the pipeline does)
    print(f"\nStep 3: Fetching stats for commits (testing first 10)...")
    success_count = 0
    fail_count = 0
    
    for i, sha in enumerate(batch_shas[:10]):
        try:
            start = time.time()
            stats_result = subprocess.run(
                ['git', 'show', '--stat', '--format=', sha],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            elapsed = time.time() - start
            success_count += 1
            if i == 0:
                print(f"  Commit {i+1}/{10}: {sha[:8]} in {elapsed:.2f}s")
        except KeyboardInterrupt as e:
            print(f"✗ INTERRUPTED at Step 3 (commit {i+1}/{10}): {e}")
            fail_count += 1
            break
        except Exception as e:
            print(f"✗ ERROR at Step 3 (commit {i+1}/{10}): {e}")
            fail_count += 1
    
    print(f"  Stats fetched: {success_count}/10 successful, {fail_count} failed")
    
    if fail_count > 0:
        return False
    
    print("\n✓ All simulation steps completed successfully!")
    return True


def run_multiple_iterations(count=3):
    """Run multiple iterations to see if it's intermittent."""
    print("="*80)
    print(f"RUNNING {count} ITERATIONS")
    print("="*80)
    
    results = []
    for i in range(count):
        print(f"\n{'='*80}")
        print(f"ITERATION {i+1}/{count}")
        print(f"{'='*80}")
        
        try:
            success = simulate_ingestion_batch()
            results.append(success)
            
            if not success:
                print(f"\n✗ Iteration {i+1} FAILED")
            else:
                print(f"\n✓ Iteration {i+1} PASSED")
        except KeyboardInterrupt:
            print(f"\n✗ Iteration {i+1} INTERRUPTED BY USER")
            results.append(False)
            break
        
        if i < count - 1:
            print("\nWaiting 2 seconds before next iteration...")
            time.sleep(2)
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✓ All iterations passed - issue is NOT reproducible in this context")
    elif passed == 0:
        print("\n✗ All iterations failed - issue is CONSISTENTLY reproducible")
    else:
        print(f"\n⚠ Issue is INTERMITTENT ({(total-passed)/total*100:.0f}% failure rate)")


if __name__ == '__main__':
    try:
        run_multiple_iterations(3)
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user (Ctrl+C)")
        sys.exit(1)
