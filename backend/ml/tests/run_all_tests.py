"""
Automated Test Runner for AgroVision ML Integration Tests.
Executes all unit and integration tests and outputs a structured summary.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_tests():
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir=str(start_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"AGROVISION ML TEST SUMMARY")
    print(f"Ran:      {result.testsRun} tests")
    print(f"Passed:   {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors:   {len(result.errors)}")
    print("=" * 60)

    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
