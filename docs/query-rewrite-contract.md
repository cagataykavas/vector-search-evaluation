# Query-rewrite contract audit

LLM-generated query rewrites can improve recall while silently changing the request. Dropping an
error code, changing a year, removing a negation, or losing an authorization/filter scope can make a
healthy retrieval metric describe the wrong question. `search/rewrite_audit.py` provides a bounded,
fail-closed admission check before rewritten queries reach BM25, dense, or hybrid retrieval.

## Contract

Each benchmark artifact binds the rewrite output to exact model, prompt, retrieval-policy, and scope
digests. Every case contains the original query, language, explicitly protected literals, optional
allowlisted structured additions, and one or more rewrites.

The audit verifies that every rewrite:

- automatically preserves structured literals and preserves declared identifiers, entities, phrases,
  and constraints as token sequences;
- preserves the number of English or Turkish negation markers while allowing equivalent markers;
- keeps the declared language and access/filter scope unchanged;
- does not introduce new URLs, emails, UUIDs, mixed alphanumeric identifiers, or numbers unless the
  benchmark explicitly allowlists them;
- stays within per-query token/growth limits and global byte, case, rewrite, character, and finding
  budgets;
- is unique after Unicode NFKC, case-folding, and whitespace normalization.

The parser rejects duplicate JSON fields, unknown fields, non-finite values, unsafe Unicode control
characters, stale/future evidence, duplicate identities, invalid digests, and protected literals that
do not actually occur in the original query.

## CLI

```bash
python -m search.rewrite_audit rewrite-artifact.json \
  --as-of 2026-09-26T23:00:00Z \
  --output rewrite-report.json
```

Exit codes are stable for CI:

- `0`: accepted;
- `2`: well-formed evidence violates rewrite policy;
- `3`: malformed/untrusted artifact or I/O failure.

Reports contain only counts, bounded reason codes, one-way case/rewrite identifiers, and canonical
artifact/policy SHA-256 digests. Raw queries, protected literals, scopes, and rewrite text are not
copied into the report. Output files are replaced atomically after flush and `fsync`.

## Trust and methodology limits

This is a structural safety contract, not a semantic-equivalence proof. Lexical negation lists can
miss paraphrases or language-specific morphology; preserving an entity does not prove that its role
was preserved; and an accepted rewrite can still retrieve worse evidence. Conversely, a safe
paraphrase can be rejected if it cannot satisfy the explicit literal contract.

The benchmark producer is trusted to label protected literals and allowed additions correctly, and
SHA-256 binding does not authenticate that producer. Production deployment should sign artifacts,
derive scope digests from the actual authorization/filter object, run this admission check before
fan-out, and then apply the existing paired robustness and relevance gates to the retrieved rankings.
