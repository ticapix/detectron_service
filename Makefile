RM=rm -rf

.phony: help

help:
	$(info Hey)

install:
	pip install --user --upgrade pip
	pip install --user -r requirements.txt

run:
	PYTHONPATH=/usr/local LD_LIBRARY_PATH=/usr/local/lib ./server.py

clean:
