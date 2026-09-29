# RAG index and encoder compatibility gate

A vector's dimension is not a sufficient serving contract. Two embedding models can emit the same
number of values while assigning them different semantics, and an index process can load different
document, metadata or embedding bytes under a familiar alias. Both failures produce plausible
scores instead of an obvious exception.

`search.index_contract` creates a content-addressed manifest directly from the repository's
`Document` objects. The manifest binds:

- pinned index and corpus revisions, rejecting common mutable aliases;
- embedding provider, model, pinned revision, dimension and vector dtype;
- distance metric and normalization location;
- chunker and tokenizer identities plus immutable revisions;
- sorted document identities, text bytes and canonical metadata;
- every embedding encoded as deterministic IEEE-754 bytes in its declared dtype; and
- document counts and exact text/metadata byte accounting.

Before serving a query, `audit_index_compatibility` recomputes the loaded index identity and checks
the query encoder and active retrieval configuration against that manifest. It rejects stale or
future-dated manifests, a mutable `latest`/`main` revision, changed corpus or vector bytes, wrong
model revision, dimension, dtype, distance metric, normalization mode, index identity or expected
manifest digest.

```python
from datetime import datetime, timezone

from search.index_contract import (
    EncoderContract,
    IndexBuildContract,
    ServingObservation,
    audit_index_compatibility,
    build_index_manifest,
)

encoder = EncoderContract(
    provider_id="internal",
    model_id="embed-v3",
    model_revision="sha256:abc123",
    dimension=1536,
)
build = IndexBuildContract(
    index_id="support-kb",
    index_revision="idx-20260929.1",
    corpus_revision="corpus-42",
    chunker_id="recursive",
    chunker_revision="chunk-7",
    tokenizer_id="cl100k_base",
    tokenizer_revision="tok-2025.1",
    distance_metric="cosine",
    normalization="engine_l2",
    built_at=datetime.now(timezone.utc),
)
manifest = build_index_manifest(documents, build=build, encoder=encoder)

report = audit_index_compatibility(
    manifest=manifest,
    expected_manifest_sha256=manifest.manifest_sha256,
    loaded_documents=engine.documents,
    observation=ServingObservation(
        index_id=build.index_id,
        index_revision=build.index_revision,
        distance_metric=build.distance_metric,
        normalization=build.normalization,
        query_encoder=encoder,
    ),
    now=datetime.now(timezone.utc),
)
assert report.accepted
```

The module also exposes a strict JSON CLI for a deployment admission or readiness hook:

```bash
python -m search.index_contract \
  --input artifacts/index-serving-observation.json \
  --now 2026-09-30T00:00:00Z \
  --output artifacts/index-compatibility-report.json
```

Exit code `0` means accepted, `2` is a well-formed compatibility-policy rejection, and `3` is
malformed or untrustworthy evidence. JSON parsing rejects duplicate fields and non-finite numbers;
input, document, dimension, text and metadata budgets are bounded. Reports contain stable reason
codes and content identities without reproducing document text, document IDs or model names.

## Trust boundary and limitations

SHA-256 proves byte identity, not who built the index. A caller that can replace both the index and
the trusted expected digest can manufacture a consistent pair. The manifest does not establish
document authority, embedding quality, relevance, freshness of individual source documents or
authorization scope. Revision labels are caller assertions; rejecting mutable aliases does not
prove that a registry enforces immutability. IEEE-754 canonicalization identifies the supplied
Python values; production vector stores should additionally bind their native serialization and
index-engine configuration.

The next increment is to sign the manifest in the indexing identity domain, store it beside the
immutable index snapshot, and require its digest in the deployment plus every retrieval trace.
