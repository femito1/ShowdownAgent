#!/usr/bin/env bash
# Stop every local Showdown server, including its worker subprocesses.
pkill -f "^node pokemon-showdown start" 2>/dev/null; pkill -f "pokemon-showdown/dist/" 2>/dev/null; true
