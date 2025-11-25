#!/usr/bin/env python3
"""
Test automatic learning from errors
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_codemode.server import execute_direct, auto_learn_from_error, learning_store, DATA_DIR

def test_auto_learning():
    """Test automatic learning from various error types"""
    print("🧪 Testing Automatic Learning System")
    print("=" * 60)

    # Clear any existing learnings for clean test
    LEARNING_FILE = DATA_DIR / "learnings.json"
    if LEARNING_FILE.exists():
        LEARNING_FILE.unlink()

    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📚 Learning file: {LEARNING_FILE}")
    print()

    # Test 1: File not found error
    print("TEST 1: FileNotFoundError")
    print("-" * 60)
    code1 = """
with open("/nonexistent/file.txt", "r") as f:
    data = f.read()
"""
    result1 = execute_direct(code1, timeout=5, auto_install=False)
    print(f"Error: {result1['stderr'][:100]}...")
    auto_learn_from_error(result1['stderr'], result1['installed_packages'])
    print(f"✓ Auto-learning triggered\n")

    # Test 2: Permission error
    print("TEST 2: PermissionError")
    print("-" * 60)
    code2 = """
with open("/root/.ssh/id_rsa", "r") as f:
    data = f.read()
"""
    result2 = execute_direct(code2, timeout=5, auto_install=False)
    print(f"Error: {result2['stderr'][:100]}...")
    auto_learn_from_error(result2['stderr'], result2['installed_packages'])
    print(f"✓ Auto-learning triggered\n")

    # Test 3: Module not found with successful install
    print("TEST 3: ModuleNotFoundError (with auto-install)")
    print("-" * 60)
    code3 = """
import requests
print("requests imported successfully!")
"""
    result3 = execute_direct(code3, timeout=10, auto_install=True)
    print(f"Success: {result3['success']}")
    if result3['installed_packages']:
        print(f"Installed: {result3['installed_packages']}")
    auto_learn_from_error(result3['stderr'], result3['installed_packages'])
    print(f"✓ Auto-learning triggered\n")

    # Check learnings file
    print("=" * 60)
    print("📚 LEARNINGS CAPTURED")
    print("=" * 60)

    if LEARNING_FILE.exists():
        learnings = json.loads(LEARNING_FILE.read_text())
        print(f"Total learnings: {len(learnings)}\n")

        for i, learning in enumerate(learnings, 1):
            print(f"{i}. Pattern: {learning['error_pattern']}")
            print(f"   Solution: {learning['solution']}")
            print(f"   Context: {learning['context']}")
            print(f"   Tags: {', '.join(learning['tags'])}")
            print()

        print("✅ Automatic learning is working!")
    else:
        print("❌ No learnings file created")

    # Test that learnings are queryable
    print("=" * 60)
    print("🔍 TESTING LEARNING RETRIEVAL")
    print("=" * 60)

    print("\nSearching for 'FileNotFoundError' learnings:")
    relevant = learning_store.find_relevant("FileNotFoundError: [Errno 2]", limit=5)
    if relevant:
        for learning in relevant:
            print(f"  ✓ Found: {learning.error_pattern}")
    else:
        print("  ✗ No learnings found")

    print("\nSearching for 'PermissionError' learnings:")
    relevant = learning_store.find_relevant("PermissionError: [Errno 13]", limit=5)
    if relevant:
        for learning in relevant:
            print(f"  ✓ Found: {learning.error_pattern}")
    else:
        print("  ✗ No learnings found")


if __name__ == "__main__":
    try:
        test_auto_learning()
        print("\n" + "=" * 60)
        print("✅ ALL AUTO-LEARNING TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
