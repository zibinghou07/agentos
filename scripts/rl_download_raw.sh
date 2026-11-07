#!/usr/bin/env bash
# Download the raw parquet files used by scripts/rl_prepare_data.py into data/raw/.
#   bash scripts/rl_download_raw.sh                          # huggingface.co
#   HF_BASE=https://hf-mirror.com bash scripts/rl_download_raw.sh   # mirror (AutoDL)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${HF_BASE:-https://huggingface.co}/datasets
mkdir -p data/raw
fetch() {  # fetch <file> <repo> <config/split>
  if [[ -s "data/raw/$1" ]]; then echo "exists: $1"; return; fi
  curl -fSL --retry 3 -o "data/raw/$1.part" "$BASE/$2/resolve/refs%2Fconvert%2Fparquet/$3/0000.parquet"
  mv "data/raw/$1.part" "data/raw/$1"
  echo "downloaded: $1"
}
fetch hotpot_train_0.parquet hotpotqa/hotpot_qa distractor/train
fetch hotpot_validation.parquet hotpotqa/hotpot_qa distractor/validation
fetch 2wiki_validation.parquet framolfese/2WikiMultihopQA default/validation
fetch musique_validation.parquet dgslibisey/MuSiQue default/validation
# Must match the sha256 values recorded in results/rl_data_report.json.
sha256sum data/raw/*.parquet 2>/dev/null || shasum -a 256 data/raw/*.parquet
