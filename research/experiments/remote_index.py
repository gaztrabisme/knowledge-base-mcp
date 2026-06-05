#!/usr/bin/env python3
"""Standalone indexing script for running on a remote GPU machine.
Reads chunk JSONs, embeds the dense vector via the shared embedder adapter
(rsync `knowledge_base/embedder.py` to ~/kb-work/embedder.py beside this script),
writes a Qdrant file-based index.

Model + output dir + collection are env-overridable so each candidate builds in
isolation (same contract as the package's config.py):
    KB_EMBEDDING_MODEL  default jina v5-small
    KB_QDRANT_DIR       default <work_dir>/qdrant
    KB_COLLECTION       default developer_knowledge
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from embedder import encode_texts, load_model, model_dim

EMBEDDING_MODEL = os.environ.get("KB_EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-small-retrieval")
EMBEDDING_DIM = model_dim(EMBEDDING_MODEL)
COLLECTION_NAME = os.environ.get("KB_COLLECTION", "developer_knowledge")

BOOKS = {
    "kagglebook_secondedition": "Kaggle Book, 2nd Edition",
    "machinelearningwithpytorchandscikit-learn": "Machine Learning with PyTorch and Scikit-Learn",
    "masteringpytorch_secondedition": "Mastering PyTorch, 2nd Edition",
    "bayesiananalysiswithpython_thirdedition": "Bayesian Analysis with Python, 3rd Edition",
    "moderntimeseriesforecastingwithpython_secondedition": "Modern Time Series Forecasting with Python, 2nd Edition",
    "causalinferenceanddiscoveryinpython": "Causal Inference and Discovery in Python",
    "deepreinforcementlearninghands-on_thirdedition": "Deep Reinforcement Learning Hands-On, 3rd Edition",
    "graphmachinelearning_secondedition": "Graph Machine Learning, 2nd Edition",
    "machinelearningengineeringwithpython_secondedition": "Machine Learning Engineering with Python, 2nd Edition",
    "scikit-learncookbook_thirdedition": "Scikit-Learn Cookbook, 3rd Edition",
    "pythonmachinelearningbyexample_fourthedition": "Python Machine Learning by Example, 4th Edition",
    "mathematicsofmachinelearning": "Mathematics of Machine Learning",
    "pandascookbook_thirdedition": "Pandas Cookbook, 3rd Edition",
    "machinelearningforalgorithmictradingsecondedition": "Machine Learning for Algorithmic Trading, 2nd Edition",
    "artificialintelligenceforcybersecurity": "Artificial Intelligence for Cybersecurity",
    "hands-onartificialintelligenceforiot_secondedition": "Hands-On AI for IoT, 2nd Edition",
    "azureai-102certificationessentials": "Azure AI-102 Certification Essentials",
    "hands-onmachinelearningwithcplusplus_secondedition": "Hands-On Machine Learning with C++, 2nd Edition",
    "production-rag-guide": "Production RAG Guide",
    # Advanced ML / Deep Learning (theory canon)
    "bishop-prml": "Pattern Recognition and Machine Learning (Bishop)",
    "probml-intro": "Probabilistic Machine Learning: An Introduction (Murphy)",
    "probml-advanced": "Probabilistic Machine Learning: Advanced Topics (Murphy)",
    "understanding-deep-learning": "Understanding Deep Learning (Prince)",
    "d2l-en": "Dive into Deep Learning",
    "lbdl": "The Little Book of Deep Learning (Fleuret)",
    "cuda-programming-guide": "NVIDIA CUDA C++ Programming Guide",
    # Databases
    "postgresql-14-internals": "PostgreSQL 14 Internals",
    "postgresql-internals-interdb": "The Internals of PostgreSQL (Suzuki)",
    "use-the-index-luke": "Use The Index, Luke — SQL Indexing",
    "database-design-2nd-ed": "Database Design, 2nd Edition",
    "redis-for-dummies": "Redis for Dummies",
    "little-redis-book": "The Little Redis Book",
    "digitalocean-redis": "DigitalOcean Redis Guide",
    # Distributed Systems
    "designing-distributed-systems": "Designing Distributed Systems",
    "distributed-systems-fun-profit": "Distributed Systems for Fun and Profit",
    # Security
    "boneh-shoup-applied-cryptography": "Applied Cryptography (Boneh & Shoup)",
    "crypto-101": "Crypto 101",
    "joy-of-cryptography": "The Joy of Cryptography",
    "owasp-developer-guide": "OWASP Developer Guide",
    "owasp-wstg-v4.2": "OWASP Web Security Testing Guide v4.2",
    "owasp-asvs-v5": "OWASP ASVS v5",
    "oauth-rfcs-book": "OAuth RFCs Book",
    "owasp-cheat-sheets": "OWASP Cheat Sheet Series",
    # Rust
    "asynchronousprogramminginrust": "Asynchronous Programming in Rust",
    "creativeprojectsforrustprogrammers": "Creative Projects for Rust Programmers",
    "designpatternsandbestpracticesinrust": "Design Patterns and Best Practices in Rust",
    "gamedevelopmentwithrustandwebassembly": "Game Development with Rust and WebAssembly",
    "hands-onconcurrencywithrust": "Hands-On Concurrency with Rust",
    "practicalsystemprogrammingforrustdevelopers": "Practical System Programming for Rust Developers",
    "rust_ebook": "Rust Programming (Nouman Azam)",
    "rustforblockchainapplicationdevelopment": "Rust for Blockchain Application Development",
    "rustforcplusplusdeveloper": "Rust for C++ Developers",
    "rustprogramminghandbook": "The Rust Programming Handbook",
    "rustwebdevelopmentwithrocket": "Rust Web Development with Rocket",
    "rustwebprogramming": "Rust Web Programming",
    "speedupyourpythonwithrust": "Speed Up Your Python with Rust",
}


def text_to_sparse_vector(text: str, sparse_model: SparseTextEmbedding) -> models.SparseVector:
    """Convert text to a sparse vector using FastEmbed's Qdrant/bm25 model."""
    results = list(sparse_model.embed([text]))
    if not results:
        return models.SparseVector(indices=[0], values=[0.0])
    sparse = results[0]
    return models.SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


def main():
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    chunks_dir = work_dir / "chunks"
    qdrant_dir = Path(os.environ.get("KB_QDRANT_DIR", str(work_dir / "qdrant")))
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dense model {EMBEDDING_MODEL} (dim {EMBEDDING_DIM}) -> {qdrant_dir} [{COLLECTION_NAME}]")
    model = load_model(EMBEDDING_MODEL)
    print("Loading sparse model Qdrant/bm25...")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    print("Models loaded.")

    client = QdrantClient(path=str(qdrant_dir))
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    batch_size = int(os.environ.get("KB_INDEX_BATCH", "48"))  # was 4 (starved the 5080); 128 OOMs Jina on 15GB
    total_indexed = 0

    for chunk_file in sorted(chunks_dir.glob("*.json")):
        slug = chunk_file.stem
        print(f"Indexing: {slug}")

        existing = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="book", match=models.MatchValue(value=slug))
            ]),
            limit=1,
        )
        if existing[0]:
            print("  Already indexed, skipping.")
            continue

        with open(chunk_file) as f:
            data = json.load(f)

        parent_map = {p["parent_id"]: p["text"] for p in data["parents"]}
        children = data["children"]
        if not children:
            print("  No children, skipping.")
            continue

        for i in range(0, len(children), batch_size):
            batch = children[i:i + batch_size]
            enriched_texts = []
            for child in batch:
                heading = child.get("metadata", {}).get("heading_path", "")
                ctx = child.get("context_text", "")
                text = child["text"]
                parts = [p for p in [heading, ctx, text] if p]
                enriched_texts.append("\n\n".join(parts))

            dense_vectors = encode_texts(model, EMBEDDING_MODEL, enriched_texts, is_query=False, batch_size=batch_size)

            points = []
            for j, child in enumerate(batch):
                parent_text = parent_map.get(child["parent_id"], "")
                sparse = text_to_sparse_vector(enriched_texts[j], sparse_model)
                meta = child.get("metadata", {})

                point = models.PointStruct(
                    id=int.from_bytes(hashlib.sha256(child["child_id"].encode()).digest()[:8], "big") >> 1,
                    vector={"": dense_vectors[j], "bm25": sparse},
                    payload={
                        "child_id": child["child_id"],
                        "parent_id": child["parent_id"],
                        "child_text": child["text"],
                        "context_text": child.get("context_text", ""),
                        "parent_text": parent_text,
                        "book": meta.get("book", slug),
                        "book_title": meta.get("book_title", BOOKS.get(slug, slug)),
                        "chapter": meta.get("chapter", ""),
                        "chapter_title": meta.get("chapter_title", ""),
                        "section": meta.get("section", ""),
                        "heading_path": meta.get("heading_path", ""),
                        "chunk_type": meta.get("chunk_type", "prose"),
                    },
                )
                points.append(point)

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_indexed += len(points)
            print(f"  {min(i + batch_size, len(children))}/{len(children)}")

    client.close()
    print(f"\nDone. {total_indexed} vectors indexed in {qdrant_dir}")


if __name__ == "__main__":
    main()
