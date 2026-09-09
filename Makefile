SHELL := /bin/sh

HOST ?= 0.0.0.0
PORT ?= 8000
ARGS ?= .

.PHONY: help install sync run serve fp build

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make install                 Install the project and dependencies (enables fp)' \
		'  make sync                    Alias for make install' \
		'  make run                     Start uvicorn with reload (default port 8000)' \
		'  make serve                   Start fserver.py (default port 8113)' \
		'  make fp ARGS="README.md"     Print fserver URLs' \
		'  make build                   Build a wheel package'

install:
	uv sync

sync: install

run:
	uv run uvicorn fserver:app --host $(HOST) --port $(PORT) --reload

serve:
	uv run python fserver.py

fp:
	uv run fp $(ARGS)

build:
	uv build
