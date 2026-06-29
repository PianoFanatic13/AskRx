import argparse
import json
import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

log = logging.getLogger(__name__)

_MODEL_ID = "BAAI/bge-large-en-v1.5"
_DEFAULT_BATCH = 64
# BGE-large supports 512 tokens total including [CLS] and [SEP], so content must be ≤ 510.
# The chunker's 450-token ceiling leaves plenty of headroom; this is a final safety net.
_TOKEN_LIMIT = 510
_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"

# dosage_form and route are not currently emitted by chunk_section(); columns will be NULL.
_INSERT = """
    INSERT INTO chunks (
        setid, drug_name, rxcui,
        loinc_code, loinc_source, section_title_path, section_type,
        chunk_text, token_count, merged, merged_title_paths,
        embedding
    ) VALUES (
        %(setid)s, %(drug_name)s, %(rxcui)s,
        %(loinc_code)s, %(loinc_source)s, %(section_title_path)s, %(section_type)s,
        %(chunk_text)s, %(token_count)s, %(merged)s, %(merged_title_paths)s,
        %(embedding)s
    )
"""


def embed_and_index(
    jsonl_path: Path,
    dsn: str,
    *,
    batch_size: int = _DEFAULT_BATCH,
    model_id: str = _MODEL_ID,
    clear: bool = False,
) -> None:
    log.info("Loading model %s ...", model_id)
    model = SentenceTransformer(model_id)

    log.info("Reading chunks from %s ...", jsonl_path)
    chunks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log.info("%d chunks loaded", len(chunks))

    oversize = [c for c in chunks if c["token_count"] > _TOKEN_LIMIT]
    if oversize:
        raise ValueError(
            f"{len(oversize)} chunks exceed the {_TOKEN_LIMIT}-token BGE-large limit. "
            f"First: setid={oversize[0]['setid']}, token_count={oversize[0]['token_count']}"
        )

    with psycopg.connect(dsn) as conn:
        register_vector(conn)

        if clear:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks")
            conn.commit()
            log.info("Cleared existing rows from chunks table")

        chunks_written = 0
        for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
            batch = chunks[start : start + batch_size]
            texts = [c["chunk_text"] for c in batch]
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

            with conn.cursor() as cur:
                for chunk, emb in zip(batch, embeddings):
                    cur.execute(_INSERT, {**chunk, "embedding": emb.tolist()})
            conn.commit()
            chunks_written += len(batch)

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
