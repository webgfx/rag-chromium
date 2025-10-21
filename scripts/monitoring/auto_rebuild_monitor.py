#!/usr/bin/env python3
"""
Auto-monitor for ingestion completion and trigger status rebuild.
Checks every 5 minutes if ingestion has stopped, then rebuilds status.json.
"""

import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def is_ingestion_running():
    """Check if any ingestion process is running."""
    try:
        result = subprocess.run(
            ['powershell', '-Command', 
             "Get-Process | Where-Object {$_.CommandLine -like '*massive_chromium_ingestion*'} | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True,
            text=True,
            timeout=10
        )
        count = int(result.stdout.strip())
        return count > 0
    except Exception as e:
        print(f"Error checking processes: {e}")
        return False

def rebuild_status():
    """Run rebuild_status.py to reconstruct status from database."""
    print(f"\n{'='*80}")
    print(f"🔄 Starting status rebuild at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'rebuild_status.py'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Status rebuild completed successfully!")
            return True
        else:
            print(f"\n❌ Status rebuild failed with exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ Status rebuild timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n❌ Error running rebuild: {e}")
        return False

def main():
    """Main monitoring loop."""
    print("="*80)
    print("🔍 Auto-Rebuild Monitor Started")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Check interval: 5 minutes")
    print(f"Action: Rebuild status.json when ingestion stops")
    print("="*80)
    print()
    
    check_count = 0
    last_was_running = None
    
    while True:
        check_count += 1
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[{current_time}] Check #{check_count}: ", end='')
        
        is_running = is_ingestion_running()
        
        if is_running:
            print("✓ Ingestion is running...")
            last_was_running = True
        else:
            print("○ No ingestion detected")
            
            # Only rebuild if we detected a transition from running to stopped
            if last_was_running is True:
                print(f"\n{'='*80}")
                print("🎯 Ingestion has STOPPED - triggering status rebuild")
                print(f"{'='*80}\n")
                
                success = rebuild_status()
                
                if success:
                    print(f"\n{'='*80}")
                    print("✅ Mission accomplished! Status has been rebuilt.")
                    print(f"{'='*80}\n")
                    print("Monitor will continue checking every 5 minutes...")
                    print("Press Ctrl+C to stop the monitor.\n")
                else:
                    print("\n⚠️  Rebuild failed, will retry next time ingestion stops.\n")
            
            last_was_running = False
        
        # Wait 5 minutes
        try:
            print(f"    Next check in 5 minutes at {datetime.fromtimestamp(time.time() + 300).strftime('%H:%M:%S')}")
            time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print("🛑 Monitor stopped by user")
            print(f"{'='*80}")
            print(f"Total checks performed: {check_count}")
            print(f"Stop time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            sys.exit(0)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
