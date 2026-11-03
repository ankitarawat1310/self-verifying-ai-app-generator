#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pip install -q -r requirements.txt
python preflight.py --config config.yaml
python train.py --config config.yaml
python evaluate.py --config config.yaml
python package_results.py --config config.yaml
