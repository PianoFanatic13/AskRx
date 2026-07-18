import argparse
import json
import logging
from pathlib import Path

import psycopg
import torch
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

log = logging.getLogger(__name__)

_MODEL_ID = "BAAI/bge-large-en-v1.5"
_DEFAULT_BATCH = 64
_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"

_INSERT = """
    INSERT INTO chunks (
        setid, drug_name, rxcui, dosage_form, route,
        loinc_code, loinc_source, section_title_path, section_type,
        chunk_text, token_count, merged, merged_title_paths,
        embedding
    ) VALUES (
        %(setid)s, %(drug_name)s, %(rxcui)s, %(dosage_form)s, %(route)s,
        %(loinc_code)s, %(loinc_source)s, %(section_title_path)s, %(section_type)s,
        %(chunk_text)s, %(token_count)s, %(merged)s, %(merged_title_paths)s,
        %(embedding)s
    )
"""


def _iter_batches(path: Path, batch_size: int):
    batch = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(json.loads(line))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def embed_and_index(
    jsonl_path: Path,
    dsn: str,
    *,
    batch_size: int = _DEFAULT_BATCH,
    model_id: str = _MODEL_ID,
    clear: bool = False,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model %s on %s ...", model_id, device)
    model = SentenceTransformer(model_id, device=device)

    total = sum(1 for ln in jsonl_path.open("r", encoding="utf-8") if ln.strip())
    log.info("Embedding %d chunks from %s", total, jsonl_path)

    with psycopg.connect(dsn) as conn:
        register_vector(conn)

        if clear:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks")
            conn.commit()
            log.info("Cleared existing rows from chunks table")

        chunks_written = 0
        total_batches = (total + batch_size - 1) // batch_size
        for batch in tqdm(_iter_batches(jsonl_path, batch_size), total=total_batches, desc="Embedding"):
            texts = [c["chunk_text"] for c in batch]
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            rows = []
            for chunk, emb in zip(batch, embeddings):
                row = {**chunk, "embedding": emb.tolist()}
                if row["merged_title_paths"] is not None:
                    row["merged_title_paths"] = Jsonb(row["merged_title_paths"])
                rows.append(row)
            with conn.cursor() as cur:
                cur.executemany(_INSERT, rows)
            conn.commit()
            chunks_written += len(batch)

        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
            cur.execute(
                "CREATE INDEX chunks_embedding_idx ON chunks"
                " USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        conn.commit()
        log.info("IVFFlat index rebuilt on %d rows", chunks_written)

    log.info("Done: %d chunks embedded and indexed", chunks_written)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed chunks from JSONL and write vectors to pgvector."
    )
    parser.add_argument("--jsonl", required=True, type=Path, help="Path to chunks.jsonl")
    parser.add_argument(
        "--dsn",
        default=_DEFAULT_DSN,
        help=f"PostgreSQL DSN (default: {_DEFAULT_DSN})",
    )
    parser.add_argument("--batch-size", default=_DEFAULT_BATCH, type=int)
    parser.add_argument("--model", default=_MODEL_ID)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="TRUNCATE the chunks table before inserting (safe to rerun from scratch)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    embed_and_index(
        jsonl_path=args.jsonl,
        dsn=args.dsn,
        batch_size=args.batch_size,
        model_id=args.model,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
