"""Containment probe (pids): how many children can the child fork? Each child only sleeps; all are killed."""
import json
import os
import signal
import sys
import time

attempts = int(sys.argv[1])
pids, error = [], None
for _ in range(attempts):
    try:
        pid = os.fork()
    except OSError as e:
        error = {"errno": e.errno, "error": e.strerror}
        break
    if pid == 0:
        time.sleep(30)
        os._exit(0)
    pids.append(pid)
print(json.dumps({"probe": "pids", "attempted": attempts, "spawned": len(pids), "error": error}), flush=True)
for pid in pids:
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except OSError:
        pass
