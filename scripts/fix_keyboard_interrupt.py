#!/usr/bin/env python3
"""
Add comprehensive exception tracking to identify where KeyboardInterrupt originates.
This modifies subprocess calls to add Windows-specific handling and detailed logging.
"""

import sys
import traceback

def add_exception_tracking():
    """Patch the ingestion code to track KeyboardInterrupt origin."""
    
    code = '''
# Insert this BEFORE the subprocess.run() call at line 866:

                self.logger.info(f"About to run git log for {len(batch_shas)} commits...")
                self.logger.debug(f"Current thread: {threading.current_thread().name}")
                self.logger.debug(f"Git command: git log --format={format_str[:20]}... --no-walk + {len(batch_shas)} SHAs")
                
                # Windows-specific subprocess handling
                import subprocess
                import sys
                
                creationflags = 0
                if sys.platform == 'win32':
                    # CREATE_NEW_PROCESS_GROUP on Windows
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                    self.logger.debug("Using CREATE_NEW_PROCESS_GROUP flag for Windows")
                
                try:
                    result = subprocess.run(
                        ['git', 'log', '--format=' + format_str, '--no-walk'] + batch_shas,
                        cwd=self.extractor.repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                        creationflags=creationflags  # Add Windows flag
                    )
                    self.logger.info(f"Git log completed successfully: {len(result.stdout):,} bytes")
                except KeyboardInterrupt as e:
                    self.logger.error(f"KeyboardInterrupt during git log!")
                    self.logger.error(f"Thread: {threading.current_thread().name}")
                    self.logger.error(f"Exception: {type(e).__name__}: {e}")
                    self.logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
                    raise
                except subprocess.TimeoutExpired as e:
                    self.logger.error(f"Git log timed out after {e.timeout}s")
                    raise
                except Exception as e:
                    self.logger.error(f"Git log failed: {type(e).__name__}: {e}")
                    self.logger.error(f"Traceback: {''.join(traceback.format_tb(e.__traceback__))}")
                    raise
'''
    
    print("PATCHING INSTRUCTIONS")
    print("="*80)
    print("\n1. Add this import at the top of massive_chromium_ingestion.py:")
    print("   import threading")
    print("   import traceback")
    print("\n2. Replace the subprocess.run() call at line 866 with this code:")
    print(code)
    print("\n3. This will:")
    print("   - Add Windows CREATE_NEW_PROCESS_GROUP flag")
    print("   - Add detailed logging before/after subprocess")
    print("   - Catch KeyboardInterrupt specifically")
    print("   - Log full traceback to identify origin")
    print("   - Show which thread is executing")


print("""
HYPOTHESIS: The KeyboardInterrupt might be coming from:

1. CUDA operations sending signals during GPU work
2. Background threads in the embedding generator
3. Signal propagation from parent process
4. Windows-specific subprocess termination behavior

SOLUTION: Add CREATE_NEW_PROCESS_GROUP flag for Windows to prevent
signal propagation from parent process to git subprocess.

This is a KNOWN ISSUE on Windows where Ctrl+C or signals sent to the
parent process get propagated to all child processes in the same
process group. Using CREATE_NEW_PROCESS_GROUP isolates the git
subprocess.

ALTERNATE SIMPLER FIX:
Just add creationflags parameter to subprocess.run():

    if sys.platform == 'win32':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        creationflags = 0
    
    result = subprocess.run(
        [...],
        creationflags=creationflags
    )
""")

if __name__ == '__main__':
    add_exception_tracking()
