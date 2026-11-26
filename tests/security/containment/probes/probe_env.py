"""Containment probe (env): is the orchestrator's canary environment variable visible to the child?"""
import json
import os
import sys

key = sys.argv[1]
print(json.dumps({"probe": "env", "visible": key in os.environ, "value": os.environ.get(key)}))
