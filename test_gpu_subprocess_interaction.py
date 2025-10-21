#!/usr/bin/env python3
"""
Test if GPU operations interfere with subprocess execution.
This simulates the actual ingestion pipeline with GPU + subprocess together.
"""

import subprocess
import time
import sys
import torch
import numpy as np
from pathlib import Path

# Add rag_system to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.embeddings.generator import EmbeddingGenerator


def test_subprocess_alone():
    """Test 1: Subprocess alone (baseline)."""
    print("\n" + "="*80)
    print("TEST 1: Subprocess alone (baseline)")
    print("="*80)
    
    repo_path = r"d:\r\cr\src"
    
    try:
        start = time.time()
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1755000', '--max-count', '100'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start
        shas = result.stdout.strip().split('\n')
        print(f"✓ Fetched {len(shas)} commits in {elapsed:.2f}s")
        return True
    except KeyboardInterrupt:
        print(f"✗ INTERRUPTED!")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_gpu_then_subprocess():
    """Test 2: GPU operation THEN subprocess."""
    print("\n" + "="*80)
    print("TEST 2: GPU operation THEN subprocess")
    print("="*80)
    
    repo_path = r"d:\r\cr\src"
    
    try:
        # GPU operation
        print("Running GPU operation...")
        generator = EmbeddingGenerator(model_name="bge-m3", batch_size=256)
        texts = [f"Test document {i}" * 50 for i in range(100)]
        
        start = time.time()
        embeddings = generator.encode_texts(texts, show_progress=False)
        elapsed = time.time() - start
        print(f"✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        
        # Small delay
        time.sleep(0.5)
        
        # Now run subprocess
        print("Running subprocess after GPU...")
        start = time.time()
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1755000', '--max-count', '100'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start
        shas = result.stdout.strip().split('\n')
        print(f"✓ Fetched {len(shas)} commits in {elapsed:.2f}s")
        
        return True
    except KeyboardInterrupt:
        print(f"✗ INTERRUPTED!")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_subprocess_then_gpu():
    """Test 3: Subprocess THEN GPU operation."""
    print("\n" + "="*80)
    print("TEST 3: Subprocess THEN GPU operation")
    print("="*80)
    
    repo_path = r"d:\r\cr\src"
    
    try:
        # Subprocess first
        print("Running subprocess...")
        start = time.time()
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', '1755000', '--max-count', '100'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start
        shas = result.stdout.strip().split('\n')
        print(f"✓ Fetched {len(shas)} commits in {elapsed:.2f}s")
        
        # Small delay
        time.sleep(0.5)
        
        # Now GPU operation
        print("Running GPU operation after subprocess...")
        generator = EmbeddingGenerator(model_name="bge-m3", batch_size=256)
        texts = [f"Test document {i}" * 50 for i in range(100)]
        
        start = time.time()
        embeddings = generator.encode_texts(texts, show_progress=False)
        elapsed = time.time() - start
        print(f"✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        
        return True
    except KeyboardInterrupt:
        print(f"✗ INTERRUPTED!")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def test_interleaved_operations():
    """Test 4: Alternating GPU and subprocess (like real pipeline)."""
    print("\n" + "="*80)
    print("TEST 4: Interleaved GPU and subprocess (like real pipeline)")
    print("="*80)
    
    repo_path = r"d:\r\cr\src"
    
    try:
        generator = EmbeddingGenerator(model_name="bge-m3", batch_size=256)
        
        for iteration in range(3):
            print(f"\nIteration {iteration + 1}/3:")
            
            # Subprocess
            print(f"  Fetching commits...")
            start = time.time()
            result = subprocess.run(
                ['git', 'rev-list', '--all', '--reverse', 
                 '--skip', str(1755000 + iteration * 100), '--max-count', '100'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            elapsed = time.time() - start
            shas = result.stdout.strip().split('\n')
            print(f"  ✓ Fetched {len(shas)} commits in {elapsed:.2f}s")
            
            # GPU operation
            print(f"  Generating embeddings...")
            texts = [f"Commit {i} iteration {iteration}" * 50 for i in range(100)]
            start = time.time()
            embeddings = generator.encode_texts(texts, show_progress=False)
            elapsed = time.time() - start
            print(f"  ✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
            
            time.sleep(0.5)
        
        print("\n✓ All iterations completed successfully")
        return True
    except KeyboardInterrupt:
        print(f"\n✗ INTERRUPTED at iteration {iteration + 1}!")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heavy_gpu_with_subprocess():
    """Test 5: Heavy GPU load with subprocess during GPU operation."""
    print("\n" + "="*80)
    print("TEST 5: Heavy GPU load with subprocess during GPU computation")
    print("="*80)
    
    repo_path = r"d:\r\cr\src"
    
    try:
        print("Loading model and preparing heavy GPU workload...")
        generator = EmbeddingGenerator(model_name="bge-m3", batch_size=500)
        
        # Generate large batch of texts (similar to real workload)
        texts = [f"Test document {i}. " * 100 for i in range(500)]
        print(f"Prepared {len(texts)} documents for embedding...")
        
        # Start GPU operation
        print("Starting GPU embedding generation...")
        start_gpu = time.time()
        embeddings = generator.encode_texts(texts, show_progress=False)
        gpu_elapsed = time.time() - start_gpu
        print(f"✓ GPU operation completed: {len(embeddings)} embeddings in {gpu_elapsed:.2f}s")
        
        # Immediately run subprocess (like the real pipeline does)
        print("Running subprocess immediately after GPU...")
        start_sub = time.time()
        result = subprocess.run(
            ['git', 'log', '--format=%H%n%an%n%ae%n%ai%n%B%n--END--', '--no-walk',
             '--skip=1755000', '--max-count=10'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        sub_elapsed = time.time() - start_sub
        commits = result.stdout.split('--END--')
        print(f"✓ Subprocess completed: {len(commits)-1} commits in {sub_elapsed:.2f}s")
        
        return True
    except KeyboardInterrupt:
        print(f"✗ INTERRUPTED!")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*80)
    print("GPU + SUBPROCESS INTERACTION TEST")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*80)
    
    results = {}
    
    try:
        results['test1'] = test_subprocess_alone()
        time.sleep(2)
        
        results['test2'] = test_gpu_then_subprocess()
        time.sleep(2)
        
        results['test3'] = test_subprocess_then_gpu()
        time.sleep(2)
        
        results['test4'] = test_interleaved_operations()
        time.sleep(2)
        
        results['test5'] = test_heavy_gpu_with_subprocess()
        
    except KeyboardInterrupt:
        print("\n\n✗ Test suite interrupted by user")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed < total:
        print("\n⚠ GPU operations may be interfering with subprocess execution!")
        print("This could be the root cause of the KeyboardInterrupt issue.")
    else:
        print("\n✓ All tests passed - GPU + subprocess interaction appears stable")
