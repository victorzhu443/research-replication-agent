"""Containment probe (egress): is there a route off the host? connect() on a UDP socket to a TEST-NET-1
address (RFC 5737, never routed on the internet) makes the kernel look up a route and sends nothing."""
import json
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("192.0.2.1", 9))
    result = {"result": "permitted", "detail": "a route exists"}
except OSError as e:
    result = {"result": "blocked", "errno": e.errno, "error": e.strerror}
finally:
    s.close()
print(json.dumps({"probe": "egress", **result}))
