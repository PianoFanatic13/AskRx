-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main chunks table: one row per text chunk from a drug label
CREATE TABLE IF NOT EXISTS chunks (
    id                  BIGSERIAL PRIMARY KEY,

    -- Label identity
    setid               TEXT NOT NULL,

    -- Drug metadata
    drug_name           TEXT NOT NULL,
    rxcui               TEXT,
    dosage_form         TEXT,
    route               TEXT,

    -- Section metadata
    loinc_code          TEXT,               -- NULL if no LOINC could be assigned
    loinc_source        TEXT CHECK (loinc_source IN ('direct', 'inherited', 'title_inferred')),
    section_title_path  TEXT[],             -- e.g. ['Dosage and Administration', 'Renal Impairment']
    section_type        TEXT CHECK (section_type IN (
                            'standard',
                            'medication_guide',
                            'ppi',
                            'otc_drug_facts',
                            'patient_counseling',
                            'clinical_pharmacology'
                        )),

    -- Chunk content
    chunk_text          TEXT NOT NULL,
    token_count         INTEGER NOT NULL,

    -- Merge info
    merged              BOOLEAN NOT NULL DEFAULT FALSE,
    merged_title_paths  JSONB,              -- NULL unless merged = TRUE

    -- Vector embedding (BGE-large-en-v1.5 = 1024 dimensions)
    embedding           vector(1024),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ANN vector search index (Inverted File Index chunking)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS chunks_fts_idx
    ON chunks
    USING GIN (to_tsvector('english', chunk_text));

-- Filtering indexes
CREATE INDEX IF NOT EXISTS chunks_rxcui_idx   ON chunks (rxcui);
CREATE INDEX IF NOT EXISTS chunks_setid_idx   ON chunks (setid);
CREATE INDEX IF NOT EXISTS chunks_loinc_idx   ON chunks (loinc_code);
