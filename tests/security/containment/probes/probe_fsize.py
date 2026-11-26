"""Containment probe (fsize): write N bytes to one file in the work dir. `started` before, `finished` after."""
import json
import sys

n = int(sys.argv[1])
print(json.dumps({"probe": "fsize", "stage": "started", "bytes": n}), flush=True)
with open("big.bin", "wb") as f:
    f.write(b"x" * n)
print(json.dumps({"probe": "fsize", "stage": "finished", "bytes": n}), flush=True)
