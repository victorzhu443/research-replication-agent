"""Containment probe (memory): allocate and touch N MiB, then hold it. `started` before, `finished` after."""
import json
import sys
import time

mib = int(sys.argv[1])
print(json.dumps({"probe": "memory", "stage": "started", "mib": mib}), flush=True)
b = bytearray(mib * 1024 * 1024)
b[::4096] = b"x" * len(b[::4096])
time.sleep(10)
print(json.dumps({"probe": "memory", "stage": "finished", "mib": mib}), flush=True)
