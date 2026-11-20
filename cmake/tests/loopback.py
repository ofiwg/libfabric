# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
"""Exercise both endpoints, completion progress, and payload verification."""
import socket
import subprocess
import sys
import tempfile
import time

with socket.socket() as reservation:
    reservation.bind(("127.0.0.1", 0))
    port = str(reservation.getsockname()[1])

args = [sys.argv[1], "-p", "tcp", "-e", "msg", "-I", "20", "-S", "4096", "-c"]
with tempfile.TemporaryFile(mode="w+") as log:
    server = subprocess.Popen(args + ["-s", "127.0.0.1", "-B", port],
                              stdout=log, stderr=subprocess.STDOUT)
    try:
        # Connecting a readiness probe would consume the benchmark connection.
        # Retry the real client if it starts before the control socket listens.
        for attempt in range(20):
            client = subprocess.run(args + ["-P", port, "127.0.0.1"],
                                    capture_output=True, text=True, timeout=20)
            if client.returncode == 0 or server.poll() is not None:
                break
            time.sleep(0.1)
        if client.returncode:
            raise RuntimeError(client.stdout + client.stderr)
        if server.wait(timeout=20):
            raise RuntimeError("pingpong server failed")
    finally:
        if server.poll() is None:
            server.kill()
        server.wait()
        log.seek(0)
        print(log.read())
