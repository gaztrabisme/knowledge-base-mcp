"""Single source of truth for dense embedding — model registry + encode idiom.

Index-time (remote GPU) and query-time (Mac) MUST encode identically or the vector
spaces mismatch. Different model families expose different query/passage idioms:

  - Jina (v3/v5):  model.encode(texts, task="retrieval.query"|"retrieval.passage")
  - Qwen3 / Gemma: model.encode_query(texts) / model.encode_document(texts)
    (last-token and mean pooling respectively; both expose the query/doc prompts)

Footgun this prevents: Qwen3 SILENTLY accepts a `task=` kwarg and ignores it — so a
naive `encode(..., task=...)` path would embed Qwen3 queries with no query prompt and
nobody would error. Keying the idiom off the model id makes that impossible.

This module is intentionally dependency-light (no .config import) so it can be rsync'd
to the GPU box alongside the standalone indexer and stay byte-identical to the package.
"""

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# model id -> {dim, idiom}. idiom selects the query/passage encode call.
# The "::raw" variants are the late-chunking experiment's consistent space: query +
# passage both raw mean-pool of transformer token embeddings (no prompt/task), so a
# naive-raw index and a late-chunk index share one query encoder. The base model is
# loaded by stripping "::raw".
EMBEDDING_REGISTRY: dict[str, dict] = {
    "jinaai/jina-embeddings-v5-text-small-retrieval": {"dim": 1024, "idiom": "jina-task"},
    "jinaai/jina-embeddings-v3": {"dim": 1024, "idiom": "jina-task"},
    "Qwen/Qwen3-Embedding-0.6B": {"dim": 1024, "idiom": "encode-qd"},
    "google/embeddinggemma-300m": {"dim": 768, "idiom": "encode-qd"},
    "google/embeddinggemma-300m::raw": {"dim": 768, "idiom": "mean-raw"},
}

_DEFAULT_IDIOM = "jina-task"
_MAX_CHARS = 8000


def base_model_id(model_id: str) -> str:
    """Strip the experiment suffix to the loadable HF id."""
    return model_id.split("::", 1)[0]


def model_dim(model_id: str) -> int:
    return EMBEDDING_REGISTRY.get(model_id, {}).get("dim", 1024)


def model_idiom(model_id: str) -> str:
    return EMBEDDING_REGISTRY.get(model_id, {}).get("idiom", _DEFAULT_IDIOM)


def resolve_device(pref: str | None = None) -> str:
    """Resolve the torch device string ("cuda" | "mps" | "cpu").

    pref defaults to $KB_DEVICE ("auto"). An explicit cuda/mps/cpu is returned as-is;
    "auto" prefers cuda, then Apple mps (guarded — older torch lacks mps backend), then
    cpu. torch is imported locally to keep this module's top level dependency-light."""
    if pref is None:
        pref = os.environ.get("KB_DEVICE", "auto")
    if pref in ("cuda", "mps", "cpu"):
        return pref

    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str) -> "SentenceTransformer":
    """Load a SentenceTransformer. trust_remote_code for Jina custom code (harmless
    for Qwen3/Gemma)."""
    from sentence_transformers import SentenceTransformer
    dev = resolve_device()
    print(f"[kb] embedding device: {dev}", file=sys.stderr)
    return SentenceTransformer(base_model_id(model_id), trust_remote_code=True, device=dev)


def raw_mean_encode(model, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Masked mean-pool of raw transformer token embeddings, L2-normalized — no
    prompt/task. Used for BOTH query and passage in the "::raw" late-chunking space so
    the spaces match the late-chunk passage path (mean-pool of a SPAN of the same raw
    token embeddings). Batched for index-time throughput."""
    import torch

    truncated = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
    out: list[list[float]] = []
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size] or [""]
        enc = model.tokenizer(batch, return_tensors="pt", truncation=True,
                              max_length=8192, padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            feats = model[0]({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
            h = feats["token_embeddings"].float()              # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(v, dim=1)
        out.extend(v.cpu().tolist())
    return out


def encode_texts(
    model, model_id: str, texts: list[str], is_query: bool, batch_size: int = 8
) -> list[list[float]]:
    """Encode texts to dense vectors using the model-appropriate query/passage idiom."""
    truncated = [t[:_MAX_CHARS] if len(t) > _MAX_CHARS else t for t in texts]
    idiom = model_idiom(model_id)

    if idiom == "jina-task":
        task = "retrieval.query" if is_query else "retrieval.passage"
        vecs = model.encode(truncated, task=task, show_progress_bar=False, batch_size=batch_size)
    elif idiom == "encode-qd":
        fn = model.encode_query if is_query else model.encode_document
        vecs = fn(truncated, show_progress_bar=False, batch_size=batch_size)
    elif idiom == "mean-raw":
        # raw mean-pool space (late-chunking experiment); query and passage identical.
        return raw_mean_encode(model, truncated)
    else:
        raise ValueError(f"unknown encode idiom {idiom!r} for {model_id!r}")

    return vecs.tolist()
