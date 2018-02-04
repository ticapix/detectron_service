RM=rm -rf

.phony: help

help:
	$(info Hey)

install:
	pip install --user --upgrade pip
	pip install --user -r requirements.txt

run:
	PYTHONPATH=/usr/local LD_LIBRARY_PATH=/usr/local/lib ./server.py

serve:
# to be used with a cron like */5 * * * * cd $HOME/dectectron_service && make serve
	PYTHONPATH=/usr/local LD_LIBRARY_PATH=/usr/local/lib ./server.py&


clean:
