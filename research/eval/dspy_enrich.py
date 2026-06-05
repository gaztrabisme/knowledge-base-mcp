"""Phase 2 — DSPy-optimize the contextual-enrichment prompt on DeepSeek.

See wiki/dspy-enrichment-plan.md. The objective is EXTRINSIC: enrichment text is
never read by anyone, it only shifts a child chunk's embedding, so we optimize for
downstream RETRIEVAL quality, gated by the Phase-1 hard golden set.

Why this design (the tricky part):
  Retrieval is a *global* metric (a chunk's enrichment only helps relative to the
  rest of the enriched corpus), but DSPy optimizers want a *per-example* metric they
  can average. Re-enriching+re-embedding the whole pool per candidate prompt is cheap
  on the LLM side (DeepSeek) but fatal on the embedding side (Jina v5, local CPU).

  Resolution: a FROZEN distractor background (~600 chunks, one child per parent,
  excluding all golden positives, enriched ONCE with the baseline prompt, embedded
  ONCE into an in-memory Qdrant — dense + BM25, DBSF, identical to production). Then
  per query-example, the DSPy program enriches only that query's positive chunk with
  the candidate prompt; we embed that ONE chunk, temp-insert it, run real fusion +
  Ettin rerank, score NDCG@10/MRR at parent granularity, remove it.

  Approximation: distractors carry baseline enrichment, the positive carries the
  candidate's. The bias is CONSTANT across candidates, so it doesn't distort which
  prompt wins. The winner is re-validated on the FULL real index in gate 2D / Phase 3.

Stages:
  python3 scripts/dspy_enrich.py --smoke                 # validate LM wiring + thinking-off
  python3 scripts/dspy_enrich.py --build-bg --pool 600   # build+cache distractor background
  python3 scripts/dspy_enrich.py --bootstrap --pool 600 --train 40 --dev 60   # validate loop
  python3 scripts/dspy_enrich.py --mipro --pool 800 --train 80 --dev 120      # full optimize
"""

import argparse
import asyncio
import json
import os
import pickle
import random
import re

import dspy
import tiktoken
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models

from knowledge_base.config import (
    BM25_MODEL, BOOKS, CHUNKS_DIR, EMBEDDING_DIM,
    RERANKER_MIN_RESULTS, RERANKER_SCORE_FLOOR,
)
from knowledge_base.enrich import CONTEXT_PROMPT, _location_clause
from knowledge_base.index import embed_texts, get_sparse_model

SEED = 42
MODEL = "deepseek-v4-flash"
API_BASE = "https://api.deepseek.com"
THINKING_OFF = {"thinking": {"type": "disabled"}}
THINKING_ON = {"thinking": {"type": "enabled"}}

GOLDEN_HARD = CHUNKS_DIR.parent / "golden" / "golden_queries_hard.jsonl"
CACHE_DIR = CHUNKS_DIR.parent / "dspy_cache"   # gitignored (data/* except data/golden/)
RESULTS_DIR = CHUNKS_DIR.parent / "golden" / "eval_runs"
MINI_COLLECTION = "dspy_mini"
TEMP_ID = 2**62  # reserved point id for the freshly-enriched positive

_enc = tiktoken.get_encoding("cl100k_base")


def _score(ev_result) -> float:
    """dspy.Evaluate returns an EvaluationResult (0-100 percentage) in 3.x; older
    builds returned a bare float. Normalize to a 0-1 mean-NDCG."""
    s = getattr(ev_result, "score", ev_result)
    return float(s) / 100.0 if float(s) > 1.0 else float(s)


# ---------------------------------------------------------------------------
# env + LM wiring
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env = CHUNKS_DIR.parent.parent / ".env"
    if env.exists() and "DEEPSEEK_API_KEY" not in os.environ:
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def make_lm(thinking: bool, max_tokens: int) -> dspy.LM:
    """DeepSeek via OpenAI passthrough (NOT native deepseek/ — brand-new model may be
    absent from LiteLLM's registry; openai/ keeps extra_body intact). Thinking defaults
    ENABLED on deepseek-v4-flash, so disabled must be explicit."""
    key = os.environ["DEEPSEEK_API_KEY"]
    return dspy.LM(
        f"openai/{MODEL}",
        api_base=API_BASE,
        api_key=key,
        extra_body=THINKING_ON if thinking else THINKING_OFF,
        temperature=1.0 if thinking else 0.3,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# signature + module
# ---------------------------------------------------------------------------

class EnrichChunk(dspy.Signature):
    """Write a short context that makes this book chunk findable by search. Name the
    specific topic/section it covers and resolve ambiguous references ("this method",
    "the algorithm above", pronouns) to the concrete entities they denote. Do not
    restate or summarize the chunk's content. The output is prepended to the chunk
    before embedding, so optimize purely for retrievability."""

    book_title: str = dspy.InputField(desc="title of the source book")
    heading_path: str = dspy.InputField(desc='section breadcrumb, e.g. "MVCC > Snapshot Isolation"')
    parent_text: str = dspy.InputField(desc="the larger passage the chunk sits within")
    child_text: str = dspy.InputField(desc="the chunk to situate")
    enrichment: str = dspy.OutputField(desc="1-2 sentences, no restating the chunk")


class Enricher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(EnrichChunk)  # Predict (not CoT): task model thinks OFF

    def forward(self, book_title, heading_path, parent_text, child_text):
        return self.gen(
            book_title=book_title, heading_path=heading_path,
            parent_text=parent_text[:4000], child_text=child_text[:2000],
        )


# ---------------------------------------------------------------------------
# data: child lookup + trainset + distractor background
# ---------------------------------------------------------------------------

def load_chunk_index() -> dict:
    """child_id -> {child_text, parent_text, parent_id, heading_path, book, book_title}."""
    idx = {}
    for f in sorted(CHUNKS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        pmap = {p["parent_id"]: p["text"] for p in data["parents"]}
        for c in data["children"]:
            md = c.get("metadata", {})
            idx[c["child_id"]] = {
                "child_text": c["text"],
                "parent_text": pmap.get(c["parent_id"], ""),
                "parent_id": c["parent_id"],
                "heading_path": md.get("heading_path", ""),
                "chapter": md.get("chapter", "") or md.get("chapter_title", ""),
                "book": md.get("book", f.stem),
                "book_title": md.get("book_title", BOOKS.get(f.stem, f.stem)),
            }
    return idx


async def baseline_enrich(client: AsyncOpenAI, sem: asyncio.Semaphore, rec: dict) -> str:
    """The current hand-written prompt (enrich.py), used to freeze the distractors."""
    prompt = CONTEXT_PROMPT.format(
        parent_text=rec["parent_text"][:4000],
        child_text=rec["child_text"][:2000],
        book_title=rec["book_title"],
        location=_location_clause({"heading_path": rec["heading_path"]}),
    )
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.3, extra_body=THINKING_OFF,
                messages=[{"role": "user", "content": prompt}],
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"    baseline_enrich error: {e}")
            return f"From {rec['book_title']}, {rec['heading_path']}."


def embed_one(text: str, task: str) -> tuple[list[float], models.SparseVector]:
    dense = embed_texts([text], task=task)[0]
    sp = list(get_sparse_model().embed([text]))[0]
    sparse = models.SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
    return dense, sparse


def emb_text(heading: str, ctx: str, child: str) -> str:
    return "\n\n".join(p for p in [heading, ctx, child] if p)


def build_background(pool: int, hardneg: int = 6) -> dict:
    """Build the frozen distractor pool, baseline-enrich + embed each, cache.

    A flat random sample makes the positive trivially win (it's the only on-topic
    chunk), so the pool is HARD-NEGATIVE-aware: for each golden positive we seed the
    pool with its siblings (same book + same chapter, else same book) — the chunks
    that actually compete with it — then fill the remainder with a stratified random
    background. One child per parent; golden positives themselves are excluded."""
    cache = CACHE_DIR / f"background_{pool}_hn{hardneg}.pkl"
    if cache.exists():
        print(f"  background cache hit: {cache.name}")
        return pickle.loads(cache.read_bytes())

    random.seed(SEED)
    rows = [json.loads(l) for l in open(GOLDEN_HARD)]
    positive_ids = {cid for r in rows for cid in r["positive_child_ids"]}
    idx = load_chunk_index()
    positive_parents = {idx[c]["parent_id"] for c in positive_ids if c in idx}

    # eligible distractors: substantive, not a positive, one child per parent
    elig_parents = set(positive_parents)
    by_book: dict[str, list[str]] = {}
    by_book_chapter: dict[tuple[str, str], list[str]] = {}
    for cid, rec in idx.items():
        if cid in positive_ids or rec["parent_id"] in elig_parents:
            continue
        t = rec["child_text"].strip()
        if re.fullmatch(r"#{0,6}\s*[\w .,:-]*", t) and len(t) < 40:
            continue
        if not (90 <= len(_enc.encode(t)) <= 250):
            continue
        elig_parents.add(rec["parent_id"])
        by_book.setdefault(rec["book"], []).append(cid)
        by_book_chapter.setdefault((rec["book"], rec["chapter"]), []).append(cid)

    chosen: set[str] = set()

    # 1) hard negatives: siblings of each positive (same book+chapter, then same book)
    for cid in positive_ids:
        if cid not in idx:
            continue
        rec = idx[cid]
        sibs = list(by_book_chapter.get((rec["book"], rec["chapter"]), []))
        random.shuffle(sibs)
        picked = [s for s in sibs if s not in chosen][:hardneg]
        if len(picked) < hardneg:
            more = [s for s in by_book.get(rec["book"], []) if s not in chosen and s not in picked]
            random.shuffle(more)
            picked += more[: hardneg - len(picked)]
        chosen.update(picked)
    n_hard = len(chosen)

    # 2) fill remainder with stratified random background
    total = sum(len(v) for v in by_book.values())
    for b, cids in by_book.items():
        k = max(1, round((pool - n_hard) * len(cids) / total)) if total else 0
        avail = [c for c in cids if c not in chosen]
        random.shuffle(avail)
        chosen.update(avail[:k])
        if len(chosen) >= pool:
            break

    chosen = list(chosen)  # hard negs kept in full; never truncated below n_hard
    random.shuffle(chosen)
    print(f"  pool={len(chosen)} ({n_hard} hard negatives + {len(chosen) - n_hard} random fill) "
          f"across {len(by_book)} books; baseline-enriching on {MODEL}...")

    _load_env()

    async def _enrich_all() -> list[str]:
        client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=API_BASE,
                             timeout=60.0, max_retries=5)
        sem = asyncio.Semaphore(50)
        return await asyncio.gather(*(baseline_enrich(client, sem, idx[c]) for c in chosen))

    ctxs = asyncio.run(_enrich_all())
    print(f"    baseline-enriched {len(ctxs)} distractors; batch-embedding (Jina v5)...")

    emb_texts = [emb_text(idx[c]["heading_path"], ctx, idx[c]["child_text"])
                 for c, ctx in zip(chosen, ctxs)]
    dense_all = embed_texts(emb_texts, task="retrieval.passage")
    sparse_all = list(get_sparse_model().embed(emb_texts))

    points = []
    for i, cid in enumerate(chosen, 1):
        rec = idx[cid]
        sp = sparse_all[i - 1]
        points.append({
            "id": i, "parent_id": rec["parent_id"], "child_id": cid,
            "dense": dense_all[i - 1],
            "sparse_idx": sp.indices.tolist(), "sparse_val": sp.values.tolist(),
        })

    result = {"pool": pool, "points": points}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(result))
    print(f"  cached background -> {cache}")
    return result


def mini_client(bg: dict) -> QdrantClient:
    """In-memory Qdrant preloaded with the frozen distractor background."""
    c = QdrantClient(":memory:")
    c.create_collection(
        collection_name=MINI_COLLECTION,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    pts = [
        models.PointStruct(
            id=p["id"],
            vector={"": p["dense"],
                    "bm25": models.SparseVector(indices=p["sparse_idx"], values=p["sparse_val"])},
            payload={"parent_id": p["parent_id"], "child_id": p["child_id"]},
        )
        for p in bg["points"]
    ]
    c.upsert(collection_name=MINI_COLLECTION, points=pts)
    return c


# ---------------------------------------------------------------------------
# the metric: score one (query, candidate-enriched positive) against the background
# ---------------------------------------------------------------------------

def _rerank_score(query: str, candidates: list[dict], positives: set[str]) -> tuple[float, int | None]:
    from knowledge_base.search import get_ranker
    ranker = get_ranker()
    docs = [c["doc"] for c in candidates]
    rankings = ranker.rank(query, docs)
    scored = []
    for item in rankings:
        c = candidates[int(item["corpus_id"])]
        scored.append({**c, "score": float(item["score"])})
    kept = [r for r in scored if r["score"] >= RERANKER_SCORE_FLOOR][:10]
    if len(kept) < RERANKER_MIN_RESULTS:
        kept = scored[:RERANKER_MIN_RESULTS]
    # dedup by parent
    seen, ranked = set(), []
    for r in kept:
        if r["parent_id"] in seen:
            continue
        seen.add(r["parent_id"])
        ranked.append(r["parent_id"])
    import math
    rels = [1 if p in positives else 0 for p in ranked[:10]]
    # exactly one positive per query → ideal DCG = positive at rank 1 = 1.0
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ndcg = dcg  # / 1.0
    hit_rank = next((i + 1 for i, p in enumerate(ranked) if p in positives), None)
    return ndcg, hit_rank


# module-level handles set by the runner so the metric closure stays picklable-free
_BG_CLIENT: QdrantClient | None = None
_CHILD_IDX: dict | None = None


def retrieval_metric(example, pred, trace=None):
    """NDCG@10 of the query against (frozen distractors + this candidate-enriched positive)."""
    enrichment = (getattr(pred, "enrichment", "") or "").strip()
    rec = _CHILD_IDX[example.child_id]
    positives = {rec["parent_id"]}

    text = emb_text(rec["heading_path"], enrichment, rec["child_text"])
    dense, sparse = embed_one(text, "retrieval.passage")

    # temp-insert the positive, fuse, fetch candidates, then remove
    _BG_CLIENT.upsert(collection_name=MINI_COLLECTION, points=[models.PointStruct(
        id=TEMP_ID,
        vector={"": dense, "bm25": models.SparseVector(indices=sparse.indices, values=sparse.values)},
        payload={"parent_id": rec["parent_id"], "child_id": example.child_id},
    )])
    try:
        q_dense = embed_texts([example.query], task="retrieval.query")[0]
        sp = list(get_sparse_model().embed([example.query]))[0]
        fused = _BG_CLIENT.query_points(
            collection_name=MINI_COLLECTION,
            prefetch=[
                models.Prefetch(query=q_dense, using="", limit=50),
                models.Prefetch(query=models.SparseVector(indices=sp.indices.tolist(),
                                                           values=sp.values.tolist()),
                                using="bm25", limit=50),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            limit=50, with_payload=True,
        ).points
    finally:
        _BG_CLIENT.delete(collection_name=MINI_COLLECTION,
                          points_selector=models.PointIdsList(points=[TEMP_ID]))

    # build rerank docs (heading + enrichment + child text mirrors search.py rerank input)
    cands = []
    for p in fused:
        cid = p.payload["child_id"]
        if cid == example.child_id:
            doc = f"{enrichment}\n{rec['child_text']}" if enrichment else rec["child_text"]
        else:
            r = _CHILD_IDX[cid]
            doc = r["child_text"]  # distractor rerank doc (ctx not needed for relative ranking)
        cands.append({"parent_id": p.payload["parent_id"], "doc": doc})

    ndcg, _ = _rerank_score(example.query, cands, positives)
    return ndcg


# ---------------------------------------------------------------------------
# trainset
# ---------------------------------------------------------------------------

def build_examples(idx: dict, n: int) -> list[dspy.Example]:
    random.seed(SEED)
    rows = [json.loads(l) for l in open(GOLDEN_HARD)]
    random.shuffle(rows)
    out = []
    for r in rows:
        cid = r["positive_child_ids"][0]
        if cid not in idx:
            continue
        rec = idx[cid]
        out.append(dspy.Example(
            book_title=rec["book_title"], heading_path=rec["heading_path"],
            parent_text=rec["parent_text"], child_text=rec["child_text"],
            query=r["query"], child_id=cid,
        ).with_inputs("book_title", "heading_path", "parent_text", "child_text"))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_smoke():
    _load_env()
    dspy.configure(lm=make_lm(thinking=False, max_tokens=200))
    idx = load_chunk_index()
    exs = build_examples(idx, 3)
    enr = Enricher()
    print(f"\n=== SMOKE: enriching 3 chunks on {MODEL} (thinking OFF) ===")
    for e in exs:
        pred = enr(book_title=e.book_title, heading_path=e.heading_path,
                   parent_text=e.parent_text, child_text=e.child_text)
        print(f"\n[{e.book_title}] {e.heading_path}")
        print(f"  query : {e.query}")
        print(f"  child : {e.child_text[:120].strip()}...")
        print(f"  ENRICH: {pred.enrichment.strip()}")
    print(f"\n  (LM history entries: {len(dspy.settings.lm.history)}; "
          f"thinking-off wiring OK if enrichments above are clean 1-2 sentences)")


def stage_build_bg(pool: int, hardneg: int):
    build_background(pool, hardneg)


def _setup_eval(pool: int, hardneg: int):
    global _BG_CLIENT, _CHILD_IDX
    _load_env()
    bg = build_background(pool, hardneg)
    _BG_CLIENT = mini_client(bg)
    _CHILD_IDX = load_chunk_index()
    return bg


def stage_bootstrap(pool: int, train: int, dev: int, hardneg: int):
    _setup_eval(pool, hardneg)
    dspy.configure(lm=make_lm(thinking=False, max_tokens=200))
    exs = build_examples(_CHILD_IDX, train + dev)
    trainset, devset = exs[:train], exs[train:train + dev]
    print(f"  trainset={len(trainset)} devset={len(devset)} pool={pool}")

    from dspy.teleprompt import BootstrapFewShot
    baseline = Enricher()

    ev = dspy.Evaluate(devset=devset, metric=retrieval_metric, num_threads=1,
                       display_progress=True, display_table=0)
    base_score = _score(ev(baseline))
    print(f"\n  BASELINE mini-harness NDCG@10 = {base_score:.4f}")

    bfs = BootstrapFewShot(metric=retrieval_metric, metric_threshold=0.8,
                           max_bootstrapped_demos=4, max_labeled_demos=0)
    compiled = bfs.compile(Enricher(), trainset=trainset)
    comp_score = _score(ev(compiled))
    print(f"\n  BOOTSTRAP mini-harness NDCG@10 = {comp_score:.4f}  (Δ {comp_score - base_score:+.4f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dspy_bootstrap.json").write_text(json.dumps(
        {"pool": pool, "train": train, "dev": dev,
         "baseline_ndcg": round(base_score, 4), "bootstrap_ndcg": round(comp_score, 4)}, indent=2))
    compiled.save(str(CACHE_DIR / "enricher_bootstrap.json"))
    print(f"  saved compiled program -> {CACHE_DIR / 'enricher_bootstrap.json'}")


def stage_mipro(pool: int, train: int, dev: int, auto: str, hardneg: int):
    _setup_eval(pool, hardneg)
    task_lm = make_lm(thinking=False, max_tokens=200)
    prompt_lm = make_lm(thinking=True, max_tokens=6000)   # instruction proposer: thinking ON
    dspy.configure(lm=task_lm)
    exs = build_examples(_CHILD_IDX, train + dev)
    trainset, devset = exs[:train], exs[train:train + dev]
    print(f"  trainset={len(trainset)} devset={len(devset)} pool={pool} auto={auto}")

    from dspy.teleprompt import MIPROv2
    ev = dspy.Evaluate(devset=devset, metric=retrieval_metric, num_threads=1,
                       display_progress=True, display_table=0)
    base_score = _score(ev(Enricher()))
    print(f"\n  BASELINE mini-harness NDCG@10 = {base_score:.4f}")

    mipro = MIPROv2(metric=retrieval_metric, prompt_model=prompt_lm, task_model=task_lm,
                    auto=auto, num_threads=1, max_labeled_demos=0, max_bootstrapped_demos=4,
                    metric_threshold=0.99, seed=SEED, verbose=False)
    compiled = mipro.compile(Enricher(), trainset=trainset, valset=devset,
                             requires_permission_to_run=False)
    comp_score = _score(ev(compiled))
    print(f"\n  MIPROv2 mini-harness NDCG@10 = {comp_score:.4f}  (Δ {comp_score - base_score:+.4f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dspy_mipro.json").write_text(json.dumps(
        {"pool": pool, "train": train, "dev": dev, "auto": auto,
         "baseline_ndcg": round(base_score, 4), "mipro_ndcg": round(comp_score, 4)}, indent=2))
    compiled.save(str(CACHE_DIR / "enricher_mipro.json"))
    print(f"  saved compiled program -> {CACHE_DIR / 'enricher_mipro.json'}")


# Careful hand-written instruction variants — the "manual iteration" DSPy must beat
# (gate 2D). Each is a distinct retrievability strategy a thoughtful engineer would try.
MANUAL_VARIANTS = {
    "baseline": EnrichChunk.__doc__,
    "query-anticipation": (
        "Anticipate the specific question a developer would search for that this chunk "
        "answers, and write 1-2 sentences of context using the vocabulary that searcher "
        "would type — including synonyms and the problem framing, not just the book's "
        "wording. Name the concrete technique, API, or concept. Do not restate the chunk."
    ),
    "entity-dense": (
        "In 1-2 sentences, name the specific concepts, APIs, error messages, parameters, "
        "and techniques this chunk addresses, and resolve any pronouns or references "
        '("this method", "the example above") to the concrete entities they denote. '
        "Maximize the number of distinct, retrievable terms. Do not restate the chunk."
    ),
    "problem-symptom": (
        "Write 1-2 sentences that (a) state the problem or symptom a developer would "
        "have that this chunk resolves, and (b) name the concrete technique or entity "
        "that resolves it, using practitioner vocabulary rather than the book's phrasing. "
        "Optimize purely for making the chunk findable by search."
    ),
}


def stage_manual(pool: int, dev: int, hardneg: int):
    _setup_eval(pool, hardneg)
    dspy.configure(lm=make_lm(thinking=False, max_tokens=200))
    # devset must match the MIPROv2 split exactly for an apples-to-apples comparison
    exs = build_examples(_CHILD_IDX, _MIPRO_TRAIN + dev)
    devset = exs[_MIPRO_TRAIN:_MIPRO_TRAIN + dev]
    ev = dspy.Evaluate(devset=devset, metric=retrieval_metric, num_threads=1,
                       display_progress=True, display_table=0)
    scores = {}
    for name, instr in MANUAL_VARIANTS.items():
        prog = Enricher()
        prog.gen = dspy.Predict(EnrichChunk.with_instructions(instr))
        s = _score(ev(prog))
        scores[name] = round(s, 4)
        print(f"\n  MANUAL[{name}] mini-harness NDCG@10 = {s:.4f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dspy_manual.json").write_text(json.dumps(
        {"pool": pool, "dev": dev, "hardneg": hardneg, "scores": scores}, indent=2))
    print(f"\n  manual variants -> {dict(sorted(scores.items(), key=lambda x: -x[1]))}")


# train split used by stage_mipro; stage_manual reuses it so devsets align
_MIPRO_TRAIN = 60

# the gate winner (Phase 2D) — shipped enrichment instruction
SHIP_VARIANT = "problem-symptom"


def make_enricher(variant: str) -> "Enricher":
    enr = Enricher()
    enr.gen = dspy.Predict(EnrichChunk.with_instructions(MANUAL_VARIANTS[variant]))
    return enr


def stage_enrich_all(book: str | None, concurrency: int, limit: int, variant: str):
    """Phase 3 production enrichment: run the validated DSPy Enricher (Envelope A) over
    all un-enriched children, writing context_text into data/chunks/*.json. Resumable."""
    _load_env()
    dspy.configure(lm=make_lm(thinking=False, max_tokens=200))
    enr = make_enricher(variant)
    print(f"  enriching with variant='{variant}' on {MODEL} (thinking OFF), "
          f"num_threads={concurrency}")

    files = [CHUNKS_DIR / f"{book}.json"] if book else sorted(CHUNKS_DIR.glob("*.json"))
    # largest files first so threads stay saturated
    files = [f for f in files if f.exists()]
    files.sort(key=lambda f: f.stat().st_size, reverse=True)

    import time
    total_enr, total_skip, t0 = 0, 0, time.monotonic()
    for i, f in enumerate(files, 1):
        data = json.loads(f.read_text())
        pmap = {p["parent_id"]: p["text"] for p in data["parents"]}
        todo = [c for c in data["children"] if not c.get("context_text")]
        skip = len(data["children"]) - len(todo)
        if limit:
            todo = todo[:limit]
        if not todo:
            total_skip += skip
            continue

        examples = [
            dspy.Example(
                book_title=c["metadata"].get("book_title", BOOKS.get(f.stem, f.stem)),
                heading_path=c["metadata"].get("heading_path", ""),
                parent_text=pmap.get(c["parent_id"], ""),
                child_text=c["text"],
            ).with_inputs("book_title", "heading_path", "parent_text", "child_text")
            for c in todo
        ]
        preds = enr.batch(examples, num_threads=concurrency, max_errors=len(examples),
                          disable_progress_bar=False)
        for c, p in zip(todo, preds):
            c["context_text"] = (getattr(p, "enrichment", "") or "").strip() or \
                f"From {c['metadata'].get('book_title','')}, {c['metadata'].get('heading_path','')}."
        f.write_text(json.dumps(data, indent=2))
        total_enr += len(todo); total_skip += skip
        dt = time.monotonic() - t0
        print(f"  [{i}/{len(files)}] {f.stem}: +{len(todo)} ({skip} done)  "
              f"| total {total_enr} in {dt/60:.1f}m ({total_enr/dt:.1f}/s)")
        if limit:
            break
    dt = time.monotonic() - t0
    print(f"\nDone. enriched {total_enr}, already-done {total_skip} in {dt/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build-bg", action="store_true")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--mipro", action="store_true")
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--enrich-all", action="store_true", help="Phase 3 production enrichment")
    ap.add_argument("--pool", type=int, default=600)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--dev", type=int, default=60)
    ap.add_argument("--hardneg", type=int, default=6, help="hard negatives seeded per positive")
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--book", default=None, help="enrich-all: single book slug (default all)")
    ap.add_argument("--concurrency", type=int, default=80, help="enrich-all: batch num_threads")
    ap.add_argument("--limit", type=int, default=0, help="enrich-all: cap children (smoke)")
    ap.add_argument("--variant", default=SHIP_VARIANT, help="enrich-all: prompt variant")
    args = ap.parse_args()

    if args.smoke:
        stage_smoke()
    elif args.build_bg:
        stage_build_bg(args.pool, args.hardneg)
    elif args.bootstrap:
        stage_bootstrap(args.pool, args.train, args.dev, args.hardneg)
    elif args.mipro:
        stage_mipro(args.pool, args.train, args.dev, args.auto, args.hardneg)
    elif args.manual:
        stage_manual(args.pool, args.dev, args.hardneg)
    elif args.enrich_all:
        stage_enrich_all(args.book, args.concurrency, args.limit, args.variant)
    else:
        ap.error("pick one of --smoke / --build-bg / --bootstrap / --mipro / --manual / --enrich-all")


if __name__ == "__main__":
    main()
