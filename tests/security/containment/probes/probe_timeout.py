"""Containment probe (timeout): spawn a grandchild, record its pid for the host, and sleep past the wall clock."""
import json
import subprocess
import sys
import time

p = subprocess.Popen(["sleep", "120"])
with open(sys.argv[1], "w") as f:
    f.write(str(p.pid))
print(json.dumps({"probe": "timeout", "stage": "started", "grandchild": p.pid}), flush=True)
time.sleep(120)
print(json.dumps({"probe": "timeout", "stage": "finished"}), flush=True)
