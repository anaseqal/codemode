#!/usr/bin/env python3
"""
Test script for streaming output functionality
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_codemode.server import execute_streaming

def test_streaming_basic():
    """Test basic streaming with delays"""
    print("=" * 60)
    print("TEST 1: Basic Streaming with Time Delays")
    print("=" * 60)

    code = """
import time

for i in range(5):
    print(f"Step {i+1}/5: Processing...")
    time.sleep(0.5)

print("✓ All steps completed!")
"""

    print("\nStreaming output:")
    print("-" * 60)

    gen = execute_streaming(code, timeout=30, auto_install=True)
    result = None

    try:
        while True:
            line = next(gen)
            print(line)
    except StopIteration as e:
        result = e.value

    print("-" * 60)
    print(f"\nResult: {result}")
    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time']:.2f}s")


def test_streaming_with_errors():
    """Test streaming with stderr output"""
    print("\n" + "=" * 60)
    print("TEST 2: Streaming with Errors/Warnings")
    print("=" * 60)

    code = """
import sys

print("Starting process...")
print("This goes to stdout", file=sys.stdout)
print("This is a warning!", file=sys.stderr)
print("Processing complete!")
"""

    print("\nStreaming output:")
    print("-" * 60)

    gen = execute_streaming(code, timeout=30, auto_install=True)
    result = None

    try:
        while True:
            line = next(gen)
            print(line)
    except StopIteration as e:
        result = e.value

    print("-" * 60)
    print(f"\nResult: {result}")
    print(f"Success: {result['success']}")


def test_streaming_with_auto_install():
    """Test streaming with package auto-install"""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming with Auto-Install (if needed)")
    print("=" * 60)

    code = """
# This should already be installed, but tests the install flow
import requests

print("✓ requests module loaded")
print(f"requests version: {requests.__version__}")
"""

    print("\nStreaming output:")
    print("-" * 60)

    gen = execute_streaming(code, timeout=30, auto_install=True)
    result = None

    try:
        while True:
            line = next(gen)
            print(line)
    except StopIteration as e:
        result = e.value

    print("-" * 60)
    print(f"\nResult: {result}")
    print(f"Success: {result['success']}")
    if result['installed_packages']:
        print(f"Installed packages: {result['installed_packages']}")


def test_streaming_progress_bar():
    """Test streaming with progress indicators"""
    print("\n" + "=" * 60)
    print("TEST 4: Streaming Progress Indicators")
    print("=" * 60)

    code = """
import time

total = 10
for i in range(total):
    progress = (i + 1) / total * 100
    bar = "█" * (i + 1) + "░" * (total - i - 1)
    print(f"Progress: [{bar}] {progress:.0f}%")
    time.sleep(0.3)

print("\\n✓ Task completed successfully!")
"""

    print("\nStreaming output:")
    print("-" * 60)

    gen = execute_streaming(code, timeout=30, auto_install=True)
    result = None

    try:
        while True:
            line = next(gen)
            print(line)
    except StopIteration as e:
        result = e.value

    print("-" * 60)
    print(f"\nResult: {result}")
    print(f"Success: {result['success']}")
    print(f"Execution time: {result['execution_time']:.2f}s")


if __name__ == "__main__":
    print("\n🧪 MCP CodeMode - Streaming Output Tests\n")

    try:
        test_streaming_basic()
        test_streaming_with_errors()
        test_streaming_with_auto_install()
        test_streaming_progress_bar()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
