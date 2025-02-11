#!/usr/bin/env python
import unittest
import logging
import sys

# Configure logging for detailed output if needed.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    # Discover all test files matching pattern 'test*.py' in the tests directory.
    suite = unittest.defaultTestLoader.discover(start_dir="tests", pattern="test*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
