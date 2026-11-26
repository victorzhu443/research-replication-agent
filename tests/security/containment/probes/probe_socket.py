"""Containment probe (socket): TCP connect to 127.0.0.1 on a port nothing listens on. Same semantics as
the toolkit grader: a refusal means the attempt reached the network stack, i.e. `permitted`."""
import json
import socket

s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
c = socket.socket()
c.settimeout(2)
try:
    c.connect(("127.0.0.1", port))
    result = {"result": "permitted", "detail": "connected"}
except ConnectionRefusedError:
    result = {"result": "permitted", "detail": "refused: the attempt reached the network stack"}
except OSError as e:
    result = {"result": "blocked", "errno": e.errno, "error": e.strerror}
finally:
    c.close()
print(json.dumps({"probe": "socket", **result}))
