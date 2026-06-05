#!/bin/bash
# v6.6 remote GPU sweep: Gemma embedder, late-chunk (Gemma raw naive/late), enrichment.
# Each candidate: index on the 5080, eval on the box (qdrant stays local), write pq-*.jsonl.
# Waits for the in-flight Qwen3 eval to free the GPU first.
set -uo pipefail
cd "$HOME/kb-work"
source "$HOME/kb-env/bin/activate"
export PYTHONPATH=. KB_COLLECTION=developer_knowledge
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

until ! pgrep -f "eval_retrieval.py --label v66-qwen3" >/dev/null; do sleep 10; done
LOG "qwen3 eval cleared; starting GPU sweep"

eval_cand(){ # qdrant_dir  query_model  label
  local qd=$1 model=$2 label=$3
  LOG "EVAL $label (query-model=$model)"
  KB_QDRANT_DIR="$HOME/kb-work/$qd" KB_EMBEDDING_MODEL="$model" PYTHONPATH=. \
    python3 -u scripts/eval_retrieval.py --label "$label" \
    --golden golden/golden_queries_hard.jsonl \
    --per-query "$HOME/kb-work/pq-$label.jsonl" --by-archetype 2>&1 | tail -16
}

# 1) Gemma embedder swap (encode-qd) — vs Jina baseline
LOG "INDEX gemma (embedder swap)"
rm -rf "$HOME/kb-work/qdrant_gemma"
KB_EMBEDDING_MODEL=google/embeddinggemma-300m KB_QDRANT_DIR="$HOME/kb-work/qdrant_gemma" KB_INDEX_BATCH=32 \
  python3 -u remote_index.py . 2>&1 | tail -3
eval_cand qdrant_gemma google/embeddinggemma-300m v66-gemma-hard

# 2) Late chunking: Gemma raw-naive (control) + raw-late (treatment); eval in ::raw space.
#    The late-chunk EFFECT = raw-late vs raw-naive (same model/space).
for mode in raw-naive raw-late; do
  LOG "INDEX lc-$mode"
  rm -rf "$HOME/kb-work/qdrant_lc_$mode"
  KB_BASE_MODEL=google/embeddinggemma-300m KB_LC_MODE=$mode KB_QDRANT_DIR="$HOME/kb-work/qdrant_lc_$mode" \
    python3 -u remote_index_lc.py . 2>&1 | tail -3
  eval_cand "qdrant_lc_$mode" "google/embeddinggemma-300m::raw" "v66-lc-$mode-hard"
done

# 3) Enrichment: Jina v5 + enriched chunks (chunks live in ~/kb-enr/chunks)
LOG "INDEX enrichment (jina v5 + enriched chunks)"
rm -rf "$HOME/kb-work/qdrant_enriched"
KB_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small-retrieval KB_QDRANT_DIR="$HOME/kb-work/qdrant_enriched" KB_INDEX_BATCH=128 \
  python3 -u remote_index.py "$HOME/kb-enr" 2>&1 | tail -3
eval_cand qdrant_enriched jinaai/jina-embeddings-v5-text-small-retrieval v66-enrich-hard

LOG "GPU SWEEP DONE"
