import sys
import os

print("Testing RAFT import...")

# Try different paths
paths_to_try = [
    'RAFT/core',
    os.path.join(os.path.dirname(__file__), 'RAFT', 'core'),
    os.path.abspath('RAFT/core')
]

for path in paths_to_try:
    print(f"\nTrying path: {path}")
    print(f"  Exists: {os.path.exists(path)}")
    
    if os.path.exists(path):
        sys.path.insert(0, path)
        try:
            from raft import RAFT
            print(f"  ✓ RAFT imported from {path}")
            print(f"  RAFT class: {RAFT}")
            break
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            sys.path.remove(path)
else:
    print("\n❌ Could not import RAFT from any path")

print("\nChecking utils...")
try:
    from utils.utils import InputPadder
    print("✓ InputPadder imported from utils.utils")
except:
    try:
        from utils import InputPadder
        print("✓ InputPadder imported from utils")
    except Exception as e:
        print(f"❌ Could not import InputPadder: {e}")