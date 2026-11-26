"""Containment probe (cwd): where does a write by relative path land?"""
import json
import os
import sys
from pathlib import Path

p = Path(sys.argv[1])
p.write_text("probe\n")
print(json.dumps({"probe": "cwd", "cwd": os.getcwd(), "wrote": str(p.resolve())}))
