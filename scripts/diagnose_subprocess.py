#!/usr/bin/env python3
"""
Diagnose the subprocess KeyboardInterrupt issue.
This script tests various subprocess configurations to identify the root cause.
"""

import subprocess
import time
import signal
import sys
import os

def test_basic_git_command():
    """Test a simple git command."""
    print("Test 1: Basic git command")
    print("-" * 80)
    
    try:
        start = time.time()
        result = subprocess.run(
            ['git', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"  Output: {result.stdout.strip()}")
        return True
    except KeyboardInterrupt as e:
        print(f"✗ KeyboardInterrupt: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_git_rev_list(repo_path, index=1700000, count=10):
    """Test git rev-list command."""
    print(f"\nTest 2: git rev-list (index={index}, count={count})")
    print("-" * 80)
    
    try:
        start = time.time()
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', str(index), '--max-count', str(count)],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start
        commits = result.stdout.strip().split('\n')
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"  Retrieved {len(commits)} commits")
        return True
    except KeyboardInterrupt as e:
        print(f"✗ KeyboardInterrupt after ~{time.time() - start:.2f}s: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_git_log_batch(repo_path, shas):
    """Test git log with multiple SHAs."""
    print(f"\nTest 3: git log --no-walk with {len(shas)} commits")
    print("-" * 80)
    
    try:
        start = time.time()
        format_str = '%H%n%an%n%ae%n%ai%n%B%n--END--'
        result = subprocess.run(
            ['git', 'log', '--format=' + format_str, '--no-walk'] + shas,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"  Output size: {len(result.stdout)} bytes")
        return True
    except KeyboardInterrupt as e:
        print(f"✗ KeyboardInterrupt after ~{time.time() - start:.2f}s: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_with_signal_ignore():
    """Test with signal.SIGINT ignored."""
    print("\nTest 4: With SIGINT ignored")
    print("-" * 80)
    
    # Save original handler
    original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        start = time.time()
        result = subprocess.run(
            ['git', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        elapsed = time.time() - start
        print(f"✓ Success in {elapsed:.2f}s")
        return True
    except KeyboardInterrupt as e:
        print(f"✗ KeyboardInterrupt (even with SIG_IGN): {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, original_handler)


def test_with_creationflags(repo_path):
    """Test with Windows-specific creationflags."""
    print("\nTest 5: With CREATE_NEW_PROCESS_GROUP flag")
    print("-" * 80)
    
    try:
        start = time.time()
        # CREATE_NEW_PROCESS_GROUP = 0x00000200
        creationflags = 0x00000200 if sys.platform == 'win32' else 0
        
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1700000', '--max-count', '10'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=creationflags
        )
        elapsed = time.time() - start
        commits = result.stdout.strip().split('\n')
        print(f"✓ Success in {elapsed:.2f}s")
        print(f"  Retrieved {len(commits)} commits")
        return True
    except KeyboardInterrupt as e:
        print(f"✗ KeyboardInterrupt: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_popen_alternative(repo_path):
    """Test using Popen instead of run."""
    print("\nTest 6: Using Popen instead of run")
    print("-" * 80)
    
    try:
        start = time.time()
        process = subprocess.Popen(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1700000', '--max-count', '10'],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=30)
        elapsed = time.time() - start
        
        if process.returncode == 0:
            commits = stdout.strip().split('\n')
            print(f"✓ Success in {elapsed:.2f}s")
            print(f"  Retrieved {len(commits)} commits")
            return True
        else:
            print(f"✗ Process failed with code {process.returncode}")
            return False
    except KeyboardInterrupt as e:
        print(f"✗ KeyboardInterrupt: {e}")
        try:
            process.kill()
        except:
            pass
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    repo_path = r"d:\r\cr\src"
    
    print("="*80)
    print("SUBPROCESS DIAGNOSTICS")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Repository: {repo_path}")
    print("="*80)
    
    results = {}
    
    # Test 1: Basic command
    results['basic'] = test_basic_git_command()
    time.sleep(1)
    
    # Test 2: git rev-list
    results['rev_list'] = test_git_rev_list(repo_path, 1700000, 10)
    time.sleep(1)
    
    # Test 3: Get some commits for batch test
    if results['rev_list']:
        print("\nPreparing for batch test...")
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1700000', '--max-count', '10'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        shas = result.stdout.strip().split('\n')[:10]
        results['batch_log'] = test_git_log_batch(repo_path, shas)
        time.sleep(1)
    
    # Test 4: Signal ignore
    results['signal_ignore'] = test_with_signal_ignore()
    time.sleep(1)
    
    # Test 5: Creation flags
    if sys.platform == 'win32':
        results['creationflags'] = test_with_creationflags(repo_path)
        time.sleep(1)
    
    # Test 6: Popen
    results['popen'] = test_popen_alternative(repo_path)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if all(results.values()):
        print("✓ All tests passed! The issue may be intermittent or context-specific.")
    else:
        failed = [name for name, success in results.items() if not success]
        print(f"✗ Failed tests: {', '.join(failed)}")
        
        if 'creationflags' in results and results['creationflags']:
            print("\n→ SOLUTION: Use CREATE_NEW_PROCESS_GROUP flag on Windows")
            print("  Add to subprocess.run(): creationflags=0x00000200")
        
        if 'popen' in results and results['popen'] and not results.get('rev_list'):
            print("\n→ SOLUTION: Use Popen instead of run")
            print("  subprocess.Popen() appears more stable")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Diagnostic script interrupted by user")
        sys.exit(1)
