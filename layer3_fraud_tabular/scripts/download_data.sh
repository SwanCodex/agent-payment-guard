#!/usr/bin/env bash
# Downloads the two IEEE-CIS files used by Layer 3A into <repo>/data/raw (git-ignored).
# Needs the Kaggle CLI with an API token, and you must accept the competition rules on kaggle.com first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ../data/raw
kaggle competitions download -c ieee-fraud-detection -p ../data/raw
cd ../data/raw
unzip -o ieee-fraud-detection.zip train_transaction.csv train_identity.csv
echo "done: $(pwd)"
