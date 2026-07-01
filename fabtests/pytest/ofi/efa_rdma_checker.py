#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.

"""
Python replacement for fi_efa_rdma_checker.

Can be used as a CLI tool (drop-in replacement for the C binary) or imported
and called directly from pytest tests.

CLI usage:
    python -m ofi.efa_rdma_checker -o read
    python -m ofi.efa_rdma_checker -o write
    python -m ofi.efa_rdma_checker -o writedata
"""

import argparse
import sys

from ofi.verbs import check_rdma_capability


def main():
    parser = argparse.ArgumentParser(description="Check EFA RDMA capabilities")
    parser.add_argument(
        "-o", dest="operation", default="read",
        choices=["read", "write", "writedata"],
        help="RDMA operation type: read | write | writedata",
    )
    args = parser.parse_args()

    try:
        enabled = check_rdma_capability(args.operation)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    if enabled:
        print(f"rdma {args.operation} is enabled")
        return 0
    else:
        print(f"rdma {args.operation} is NOT enabled", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
