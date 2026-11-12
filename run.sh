#!/bin/sh
set -e
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python run.py "$@"
