.PHONY: help venv install smoke-sts smoke-s3 bootstrap-s3

VENV_PY := .venv/bin/python3
PIP := .venv/bin/pip3
DEPS_SENTINEL := .venv/.deps_installed

help:
	@echo "Targets:"
	@echo "  make venv          Create .venv"
	@echo "  make install       Install Python deps into .venv"
	@echo "  make smoke-sts     Validate AWS identity (STS)"
	@echo "  make smoke-s3      Validate S3 bucket + prefix listing"
	@echo "  make bootstrap-s3  Create S3 'folder marker' prefixes"

$(VENV_PY):
	python3 -m venv .venv
	$(PIP) install --upgrade pip

venv: $(VENV_PY)

$(DEPS_SENTINEL): $(VENV_PY) requirements.txt
	$(PIP) install -r requirements.txt
	@touch $(DEPS_SENTINEL)

install: $(DEPS_SENTINEL)

smoke-sts: install
	$(VENV_PY) -m src.smoke.smoke_sts

smoke-s3: install
	$(VENV_PY) -m src.smoke.smoke_s3

bootstrap-s3: install
	$(VENV_PY) -m src.smoke.bootstrap_s3_prefixes
