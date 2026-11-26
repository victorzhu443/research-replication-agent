SANDBOX_IMAGE ?= replicator-sandbox:latest

.PHONY: sandbox-image security-tests test

# Build the container the sandbox runs generated code in; `auto` picks it up once it exists.
sandbox-image:
	docker build -f Dockerfile.sandbox -t $(SANDBOX_IMAGE) .

security-tests:
	.venv/bin/python -m pytest tests/security/ -q

test:
	.venv/bin/python -m pytest tests/ -q
