"""Containment probe (abs-write): can the child write by absolute path, and by `..`, outside the work dir?"""
import json
import sys
from pathlib import Path

out = {"probe": "abs-write"}
for label, target in (("absolute", sys.argv[1]), ("dotdot", sys.argv[2])):
    try:
        Path(target).write_text("probe\n")
        out[label] = {"wrote": True}
    except OSError as e:
        out[label] = {"wrote": False, "errno": e.errno, "error": e.strerror}
print(json.dumps(out))
