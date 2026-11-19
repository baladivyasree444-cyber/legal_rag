# Legal Document RAG Practice Dataset

Synthetic-only dataset for practicing legal RAG. No real people, companies, contracts, or legal advice are intended.

## Contents
- 8 synthetic contracts
- metadata.json
- test_queries.json

## Recommended RAG experiments
1. Parse each document and preserve Document ID, section number, and metadata.
2. Chunk by legal section rather than arbitrary character boundaries.
3. Build dense-vector retrieval.
4. Build BM25 retrieval.
5. Combine the rankings using Reciprocal Rank Fusion (RRF).
6. Apply metadata filters such as jurisdiction, document type, version, and party.
7. Rerank the top candidates.
8. Generate answers only from retrieved clauses.
9. Return citations as `Document ID -> Section`.
10. For comparison, retrieve both documents and align clauses by section/topic.

## Intentional differences
EMP-001 vs EMP-002: salary, bonus, confidentiality duration, post-employment restriction, termination notice, severance, jurisdiction.
NDA-001 vs NDA-002: mutual vs one-way, permitted use, security incident deadline, duration, return deadline, jurisdiction.
SVC-001 vs SVC-002: price, payment window, SLA, incident deadline, IP license, liability period, cure period, convenience termination, jurisdiction.
VND-001 vs VND-002: price-adjustment period, delivery obligation, inspection window, warranty, insurance, liability period, cure period, convenience termination.

## Expected citation format
`[SVC-002, Section 5 - SECURITY INCIDENT]`
