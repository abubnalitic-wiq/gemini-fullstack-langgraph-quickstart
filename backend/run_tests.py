#!/usr/bin/env python3
"""Test runner script for the SQL query executor tests.

Usage:
    python run_tests.py                    # Run all tests except integration
    python run_tests.py --integration     # Run integration tests
    python run_tests.py --all            # Run all tests
    python run_tests.py --coverage       # Run with coverage
    python run_tests.py --unit           # Run only unit tests
    python run_tests.py --verbose        # Run with verbose output
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: list, description: str = ""):
    """Run a command and handle errors"""
    print(f"🔄 {description}")
    print(f"   Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run SQL query executor tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all tests including integration"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage reporting"
    )
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--test", type=str, help="Run specific test")

    args = parser.parse_args()

    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])
    else:
        pytest_cmd.extend(["-v"])

    # Add coverage
    if args.coverage:
        pytest_cmd.extend(
            [
                "--cov=src/research_agent/database",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
        )

    # Determine test selection
    if args.integration:
        pytest_cmd.extend(["-m", "integration"])
        description = "Running integration tests"
    elif args.all:
        pytest_cmd.extend(["-m", ""])
        description = "Running all tests"
    elif args.unit:
        pytest_cmd.extend(["-m", "not integration"])
        description = "Running unit tests only"
    elif args.file:
        pytest_cmd.append(args.file)
        description = f"Running tests from {args.file}"
    elif args.test:
        pytest_cmd.extend(["-k", args.test])
        description = f"Running test matching: {args.test}"
    else:
        # Default: run all tests except integration
        pytest_cmd.extend(["-m", "not integration"])
        description = "Running all tests except integration"

    # Add test directory
    pytest_cmd.append("tests/")

    # Run the tests
    success = run_command(pytest_cmd, description)

    if success:
        print("\n✅ All tests passed!")

        # Show coverage report location if coverage was run
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
