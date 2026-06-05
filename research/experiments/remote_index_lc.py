#!/usr/bin/env python3
"""Standalone late-chunking / raw-mean indexer for the Stage 2 experiment.

Modes (env KB_LC_MODE):
  raw-naive : children embedded in ISOLATION via raw mean-pool (no prompt). CONTROL.
  raw-late  : children embedded as SPANS of the parent's raw token embeddings. Late
              chunking in the raw-mean space (Gemma — last-token base, but its ST
              pooling is mean, so raw mean-pool is valid). TREATMENT.
  jina-late : children as spans of the parent's NATIVE retrieval token embeddings
              (Jina v3 task-LoRA, task=retrieval.passage). Faithful native late chunking.

The late-chunk effect is (late - naive) for the SAME model/space:
  Gemma : raw-late  vs raw-naive
  Jina3 : jina-late vs jina-naive (jina-naive = ordinary remote_index.py with v3)

BM25 sparse stays on heading+child text (model-independent) so fusion is comparable.
Dense is the only thing late chunking changes.

Env: KB_BASE_MODEL, KB_LC_MODE, KB_QDRANT_DIR, KB_COLLECTION (default developer_knowledge).
rsync embedder.py beside this script. Run inside the env that loads the base model
(kb-env for Gemma, kb-env-v3 for Jina v3).
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from embedder import load_model, model_dim, raw_mean_encode

BASE_MODEL = os.environ["KB_BASE_MODEL"]
LC_MODE = os.environ["KB_LC_MODE"]
COLLECTION = os.environ.get("KB_COLLECTION", "developer_knowledge")
EMBEDDING_DIM = model_dim(BASE_MODEL)


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _map_norm_span(parent, norm_start, norm_len):
    out_start = out_end = None
    norm_i = 0
    prev_space = False
    for oi, ch in enumerate(parent):
        is_space = ch.isspace()
        if is_space and prev_space:
            prev_space = True
            continue
        if norm_i == norm_start and out_start is None:
            out_start = oi
        if norm_i == norm_start + norm_len:
            out_end = oi
            break
        norm_i += 1
        prev_space = is_space
    if out_start is None:
        return None
    return out_start, out_end if out_end else len(parent)


def _find_span(parent, child):
    i = parent.find(child)
    if i >= 0:
        return i, i + len(child)
    pn = re.sub(r"\s+", " ", parent)
    cn = re.sub(r"\s+", " ", child).strip()
    j = pn.find(cn)
    if j < 0:
        return None
    return _map_norm_span(parent, j, len(cn))


def _token_embeddings(model, text):
    """(hidden [T,H] np, offsets) for the parent. raw transformer forward for raw-*,
    native task-LoRA passage forward for jina-late."""
    if LC_MODE == "jina-late":
        hidden = model.encode(text, task="retrieval.passage", output_value="token_embeddings")
        hidden = hidden.detach().cpu().float().numpy() if hasattr(hidden, "detach") else np.asarray(hidden, dtype="float32")
        enc = model.tokenizer(text, return_offsets_mapping=True, truncation=True,
                              max_length=8192, add_special_tokens=True)
        offsets = enc["offset_mapping"]
    else:  # raw-late
        enc = model.tokenizer(text, return_offsets_mapping=True, return_tensors="pt",
                              truncation=True, max_length=8192, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ein = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            feats = model[0]({"input_ids": ein["input_ids"], "attention_mask": ein["attention_mask"]})
            hidden = feats["token_embeddings"][0].cpu().float().numpy()
    return hidden, offsets


def _naive_vec(model, child_text):
    if LC_MODE == "jina-late":
        v = model.encode(child_text, task="retrieval.passage")
        return _norm(np.asarray(v)).tolist()
    return raw_mean_encode(model, [child_text])[0]


def late_children(model, parent_text, child_texts):
    hidden, offsets = _token_embeddings(model, parent_text)
    if hidden.shape[0] != len(offsets):
        # alignment guard: fall back to naive for the whole parent
        return [(_naive_vec(model, c), False) for c in child_texts]
    out = []
    for ct in child_texts:
        span = _find_span(parent_text, ct)
        if span is None:
            out.append((_naive_vec(model, ct), False))
            continue
        a, b = span
        idx = [i for i, off in enumerate(offsets)
               if off[1] > a and off[0] < b and not (off[0] == 0 and off[1] == 0)]
        if not idx:
            out.append((_naive_vec(model, ct), False))
            continue
        out.append((_norm(hidden[idx].mean(axis=0)).tolist(), True))
    return out


def sparse_vec(text, sparse_model):
    res = list(sparse_model.embed([text]))
    if not res:
        return models.SparseVector(indices=[0], values=[0.0])
    s = res[0]
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def main():
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    chunks_dir = work_dir / "chunks"
    qdrant_dir = Path(os.environ.get("KB_QDRANT_DIR", str(work_dir / "qdrant")))
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{LC_MODE}] base={BASE_MODEL} dim={EMBEDDING_DIM} -> {qdrant_dir} [{COLLECTION}]")
    model = load_model(BASE_MODEL)
    sparse_model = SparseTextEmbedding("Qdrant/bm25")

    client = QdrantClient(path=str(qdrant_dir))
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    total, found_total, span_total = 0, 0, 0
    for chunk_file in sorted(chunks_dir.glob("*.json")):
        slug = chunk_file.stem
        data = json.loads(chunk_file.read_text())
        parent_map = {p["parent_id"]: p["text"] for p in data["parents"]}
        children = data["children"]
        if not children:
            continue

        by_parent = {}
        for c in children:
            by_parent.setdefault(c["parent_id"], []).append(c)

        points = []
        for pid, kids in by_parent.items():
            ptext = parent_map.get(pid, "")
            child_texts = [c["text"] for c in kids]

            if LC_MODE == "raw-naive":
                dense = raw_mean_encode(model, child_texts)
                flags = [True] * len(child_texts)
            else:
                pairs = late_children(model, ptext, child_texts)
                dense = [v for v, _ in pairs]
                flags = [f for _, f in pairs]

            for c, dv, f in zip(kids, dense, flags):
                meta = c.get("metadata", {})
                heading = meta.get("heading_path", "")
                enriched = "\n\n".join(p for p in [heading, c.get("context_text", ""), c["text"]] if p)
                span_total += 1
                found_total += 1 if f else 0
                points.append(models.PointStruct(
                    id=int.from_bytes(hashlib.sha256(c["child_id"].encode()).digest()[:8], "big") >> 1,
                    vector={"": dv, "bm25": sparse_vec(enriched, sparse_model)},
                    payload={
                        "child_id": c["child_id"], "parent_id": c["parent_id"],
                        "child_text": c["text"], "context_text": c.get("context_text", ""),
                        "parent_text": ptext, "book": meta.get("book", slug),
                        "book_title": meta.get("book_title", slug),
                        "chapter": meta.get("chapter", ""), "chapter_title": meta.get("chapter_title", ""),
                        "section": meta.get("section", ""), "heading_path": heading,
                        "chunk_type": meta.get("chunk_type", "prose"),
                    },
                ))
            if len(points) >= 256:
                client.upsert(collection_name=COLLECTION, points=points)
                total += len(points)
                points = []
        if points:
            client.upsert(collection_name=COLLECTION, points=points)
            total += len(points)
        print(f"  {slug}: {total} pts")

    client.close()
    pct = 100 * found_total / span_total if span_total else 0
    print(f"\nDone. {total} vectors. span-found {found_total}/{span_total} ({pct:.0f}%) in {qdrant_dir}")


if __name__ == "__main__":
    main()
