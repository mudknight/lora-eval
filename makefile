.PHONY: help clean build install dev-install test lint format run

# Project configuration
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SPEC_FILE := lora-eval.spec
DIST_DIR := dist
BUILD_DIR := build
INSTALL_DIR := $(HOME)/.local/bin
APP_NAME := lora-eval

help:
	@echo "Available targets:"
	@echo "  dep-install  - Install dependencies"
	@echo "  venv-setup   - Set up virtualenv"
	@echo "  build        - Build executable with PyInstaller"
	@echo "  install       - Symlink bin to ~/.local/bin"
	@echo "  clean        - Remove build artifacts"
	@echo "  run          - Run the application"

dep-install:
	$(PIP) install -r requirements.txt

venv-setup:
	python -m venv $(VENV)
	dep-install

build:
	$(VENV)/bin/pyinstaller $(SPEC_FILE) --noconfirm

install:
	@mkdir -p $(INSTALL_DIR)
	ln -sf $(CURDIR)/$(DIST_DIR)/$(APP_NAME) $(INSTALL_DIR)/$(APP_NAME)
	@echo "Linked $(APP_NAME) to $(INSTALL_DIR)"

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.spec~" -delete

run:
	$(PYTHON) main.py
