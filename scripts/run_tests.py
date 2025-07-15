#!/usr/bin/env python3
"""Test runner script for YoloCAM library."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type='all', verbose=False, coverage=False, parallel=False):
    """Run tests with specified options.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        parallel: Run tests in parallel
    """
    # Base pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add test selection
    if test_type == 'unit':
        cmd.extend(['tests/unit'])
    elif test_type == 'integration':
        cmd.extend(['tests/integration'])
    elif test_type == 'all':
        cmd.extend(['tests/'])
    else:
        cmd.extend([f'tests/{test_type}'])
    
    # Add options
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    if coverage:
        cmd.extend(['--cov=yolocam', '--cov-report=term-missing', '--cov-report=html'])
    
    if parallel:
        cmd.extend(['-n', 'auto'])
    
    # Add markers for different test types
    if test_type == 'unit':
        cmd.extend(['-m', 'unit'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration'])
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run YoloCAM tests')
    parser.add_argument(
        'test_type', 
        nargs='?', 
        default='all',
        choices=['unit', 'integration', 'all'],
        help='Type of tests to run (default: all)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Enable coverage reporting'
    )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run only fast tests (exclude slow marker)'
    )
    
    args = parser.parse_args()
    
    # Add fast option
    if args.fast:
        # This would need to be added to the pytest command
        pass
    
    exit_code = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()