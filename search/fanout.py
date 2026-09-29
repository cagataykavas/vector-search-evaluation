from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
from collections import defaultdict
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from search.engine import HybridSearchEngine, SearchHit

VARIANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
MAX_QUERY_BYTES = 4_096
MAX_VARIANTS = 16
MAX_CONCURRENCY = 16
MAX_CANDIDATES = 1_000


class FanoutError(RuntimeError):
    """Base error with a stable, non-sensitive reason code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FanoutInputError(FanoutError):
    """The request or policy is malformed."""


class FanoutAdmissionError(FanoutError):
    """Retrieval did not satisfy the configured success policy."""


@dataclass(frozen=True)
class QueryVariant:
    variant_id: str
    query_text: str
    query_embedding: tuple[float, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class FanoutPolicy:
    top_k: int = 10
    candidate_k: int = 50
    rrf_constant: int = 60
    max_queries: int = 8
    max_concurrency: int = 4
    deadline_ms: int = 500
    min_successful_queries: int = 1
    require_primary: bool = True


@dataclass(frozen=True)
class QueryEvidence:
    variant_id: str
    query_digest: str
    status: Literal["ok", "error", "timeout"]
    hit_count: int
    reason_code: str | None = None


@dataclass(frozen=True)
class RankContribution:
    variant_id: str
    source_rank: int
    weighted_rrf: float


@dataclass(frozen=True)
class FusedHit:
    doc_id: str
    score: float
    rank: int
    source: str
    contributions: tuple[RankContribution, ...]


@dataclass(frozen=True)
class FanoutResult:
    hits: tuple[FusedHit, ...]
    evidence: tuple[QueryEvidence, ...]
    request_digest: str
    successful_queries: int
    failed_queries: int
    timed_out_queries: int


class Retriever(Protocol):
    def __call__(
        self,
        variant: QueryVariant,
        candidate_k: int,
    ) -> Sequence[SearchHit] | Awaitable[Sequence[SearchHit]]: ...


@dataclass(frozen=True)
class _Outcome:
    index: int
    variant: QueryVariant
    status: Literal["ok", "error", "timeout"]
    hits: tuple[SearchHit, ...]
    reason_code: str | None


def _normalize_query(text: str) -> str:
    return " ".join(text.split()).casefold()


def _query_digest(variant: QueryVariant) -> str:
    payload = {
        "embedding": list(variant.query_embedding),
        "query_text": variant.query_text,
        "variant_id": variant.variant_id,
        "weight": variant.weight,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_digest(variants: Sequence[QueryVariant], policy: FanoutPolicy) -> str:
    payload = {
        "policy": {
            "candidate_k": policy.candidate_k,
            "deadline_ms": policy.deadline_ms,
            "max_concurrency": policy.max_concurrency,
            "max_queries": policy.max_queries,
            "min_successful_queries": policy.min_successful_queries,
            "require_primary": policy.require_primary,
            "rrf_constant": policy.rrf_constant,
            "top_k": policy.top_k,
        },
        "queries": [_query_digest(variant) for variant in variants],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_policy(policy: FanoutPolicy) -> None:
    if not 1 <= policy.top_k <= MAX_CANDIDATES:
        raise FanoutInputError("invalid_top_k")
    if not policy.top_k <= policy.candidate_k <= MAX_CANDIDATES:
        raise FanoutInputError("invalid_candidate_k")
    if not 1 <= policy.rrf_constant <= 10_000:
        raise FanoutInputError("invalid_rrf_constant")
    if not 1 <= policy.max_queries <= MAX_VARIANTS:
        raise FanoutInputError("invalid_max_queries")
    if not 1 <= policy.max_concurrency <= MAX_CONCURRENCY:
        raise FanoutInputError("invalid_max_concurrency")
    if not 10 <= policy.deadline_ms <= 30_000:
        raise FanoutInputError("invalid_deadline_ms")
    if not 1 <= policy.min_successful_queries <= policy.max_queries:
        raise FanoutInputError("invalid_min_successful_queries")


def _validate_variants(
    variants: Sequence[QueryVariant],
    policy: FanoutPolicy,
) -> tuple[QueryVariant, ...]:
    materialized = tuple(variants)
    if not materialized:
        raise FanoutInputError("missing_queries")
    if len(materialized) > policy.max_queries:
        raise FanoutInputError("query_budget_exceeded")
    if policy.min_successful_queries > len(materialized):
        raise FanoutInputError("unreachable_success_policy")

    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    dimension: int | None = None
    for variant in materialized:
        if not isinstance(variant, QueryVariant):
            raise FanoutInputError("invalid_query_variant")
        if not isinstance(variant.variant_id, str) or not VARIANT_ID_RE.fullmatch(
            variant.variant_id
        ):
            raise FanoutInputError("invalid_variant_id")
        if variant.variant_id in seen_ids:
            raise FanoutInputError("duplicate_variant_id")
        seen_ids.add(variant.variant_id)

        if not isinstance(variant.query_text, str) or not variant.query_text.strip():
            raise FanoutInputError("empty_query")
        if len(variant.query_text.encode("utf-8")) > MAX_QUERY_BYTES:
            raise FanoutInputError("query_too_large")
        normalized = _normalize_query(variant.query_text)
        if normalized in seen_queries:
            raise FanoutInputError("duplicate_query")
        seen_queries.add(normalized)

        if not isinstance(variant.query_embedding, tuple) or not variant.query_embedding:
            raise FanoutInputError("empty_embedding")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in variant.query_embedding
        ):
            raise FanoutInputError("non_finite_embedding")
        if dimension is None:
            dimension = len(variant.query_embedding)
        elif len(variant.query_embedding) != dimension:
            raise FanoutInputError("embedding_dimension_mismatch")
        if (
            not isinstance(variant.weight, (int, float))
            or isinstance(variant.weight, bool)
            or not math.isfinite(variant.weight)
            or not 0 < variant.weight <= 10
        ):
            raise FanoutInputError("invalid_query_weight")

    return materialized


def _validate_hits(hits: Iterable[SearchHit], candidate_k: int) -> tuple[SearchHit, ...]:
    materialized = tuple(hits)
    if len(materialized) > candidate_k:
        raise FanoutError("retriever_candidate_budget_exceeded")

    seen_ids: set[str] = set()
    for expected_rank, hit in enumerate(materialized, start=1):
        if not isinstance(hit, SearchHit):
            raise FanoutError("invalid_retriever_hit")
        if (
            not isinstance(hit.doc_id, str)
            or not hit.doc_id
            or len(hit.doc_id.encode("utf-8")) > 512
        ):
            raise FanoutError("invalid_document_id")
        if any(ord(character) < 32 for character in hit.doc_id):
            raise FanoutError("invalid_document_id")
        if hit.doc_id in seen_ids:
            raise FanoutError("duplicate_document_id")
        seen_ids.add(hit.doc_id)
        if not isinstance(hit.rank, int) or isinstance(hit.rank, bool) or hit.rank != expected_rank:
            raise FanoutError("invalid_retriever_rank")
        if (
            not isinstance(hit.score, (int, float))
            or isinstance(hit.score, bool)
            or not math.isfinite(hit.score)
        ):
            raise FanoutError("non_finite_retriever_score")
        if not isinstance(hit.source, str) or not hit.source:
            raise FanoutError("missing_retriever_source")
    return materialized


def _is_async_callable(retriever: Retriever) -> bool:
    return inspect.iscoroutinefunction(retriever) or inspect.iscoroutinefunction(
        type(retriever).__call__
    )


async def _invoke_retriever(
    retriever: Retriever,
    variant: QueryVariant,
    candidate_k: int,
    deadline: float,
) -> Sequence[SearchHit]:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError
    if _is_async_callable(retriever):
        maybe_hits = retriever(variant, candidate_k)
    else:
        maybe_hits = await asyncio.wait_for(
            asyncio.to_thread(retriever, variant, candidate_k),
            timeout=remaining,
        )
    if inspect.isawaitable(maybe_hits):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(maybe_hits, timeout=remaining)
    return maybe_hits


async def _retrieve_one(
    *,
    index: int,
    variant: QueryVariant,
    retriever: Retriever,
    candidate_k: int,
    semaphore: asyncio.Semaphore,
    deadline: float,
) -> _Outcome:
    try:
        async with semaphore:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return _Outcome(index, variant, "timeout", (), "deadline_exceeded")
            raw_hits = await _invoke_retriever(retriever, variant, candidate_k, deadline)
            hits = _validate_hits(raw_hits, candidate_k)
            return _Outcome(index, variant, "ok", hits, None)
    except TimeoutError:
        return _Outcome(index, variant, "timeout", (), "deadline_exceeded")
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - adapter failures become bounded reason codes.
        code = error.code if isinstance(error, FanoutError) else "retriever_error"
        return _Outcome(index, variant, "error", (), code)


def _fuse(outcomes: Sequence[_Outcome], policy: FanoutPolicy) -> tuple[FusedHit, ...]:
    scores: defaultdict[str, float] = defaultdict(float)
    contributions: defaultdict[str, list[RankContribution]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.status != "ok":
            continue
        for hit in outcome.hits:
            contribution = outcome.variant.weight / (policy.rrf_constant + hit.rank)
            scores[hit.doc_id] += contribution
            contributions[hit.doc_id].append(
                RankContribution(
                    variant_id=outcome.variant.variant_id,
                    source_rank=hit.rank,
                    weighted_rrf=contribution,
                )
            )

    ordered = sorted(
        scores,
        key=lambda doc_id: (
            -scores[doc_id],
            min(item.source_rank for item in contributions[doc_id]),
            -len(contributions[doc_id]),
            doc_id,
        ),
    )[: policy.top_k]
    return tuple(
        FusedHit(
            doc_id=doc_id,
            score=scores[doc_id],
            rank=rank,
            source="query_fanout_rrf",
            contributions=tuple(contributions[doc_id]),
        )
        for rank, doc_id in enumerate(ordered, start=1)
    )


async def retrieve_with_query_fanout(
    variants: Sequence[QueryVariant],
    retriever: Retriever,
    *,
    policy: FanoutPolicy | None = None,
) -> FanoutResult:
    """Run bounded multi-query retrieval and deterministically fuse the rankings.

    The first variant is the primary query. With ``require_primary=True``, a timeout or
    failure on that path rejects the whole request even if an expansion succeeds.
    """

    effective_policy = policy or FanoutPolicy()
    _validate_policy(effective_policy)
    admitted = _validate_variants(variants, effective_policy)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + effective_policy.deadline_ms / 1_000
    semaphore = asyncio.Semaphore(effective_policy.max_concurrency)
    tasks = [
        asyncio.create_task(
            _retrieve_one(
                index=index,
                variant=variant,
                retriever=retriever,
                candidate_k=effective_policy.candidate_k,
                semaphore=semaphore,
                deadline=deadline,
            ),
            name=f"retrieval-{variant.variant_id}",
        )
        for index, variant in enumerate(admitted)
    ]

    done, pending = await asyncio.wait(tasks, timeout=effective_policy.deadline_ms / 1_000)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    outcomes = [task.result() for task in done]
    completed_indexes = {outcome.index for outcome in outcomes}
    outcomes.extend(
        _Outcome(index, admitted[index], "timeout", (), "deadline_exceeded")
        for index in range(len(admitted))
        if index not in completed_indexes
    )
    outcomes.sort(key=lambda outcome: outcome.index)

    if effective_policy.require_primary and outcomes[0].status != "ok":
        raise FanoutAdmissionError("primary_query_failed")
    successful = sum(outcome.status == "ok" for outcome in outcomes)
    if successful < effective_policy.min_successful_queries:
        raise FanoutAdmissionError("insufficient_successful_queries")

    evidence = tuple(
        QueryEvidence(
            variant_id=outcome.variant.variant_id,
            query_digest=_query_digest(outcome.variant),
            status=outcome.status,
            hit_count=len(outcome.hits),
            reason_code=outcome.reason_code,
        )
        for outcome in outcomes
    )
    return FanoutResult(
        hits=_fuse(outcomes, effective_policy),
        evidence=evidence,
        request_digest=_request_digest(admitted, effective_policy),
        successful_queries=successful,
        failed_queries=sum(outcome.status == "error" for outcome in outcomes),
        timed_out_queries=sum(outcome.status == "timeout" for outcome in outcomes),
    )


def hybrid_engine_retriever(engine: HybridSearchEngine) -> Retriever:
    """Adapt the repository's synchronous engine to the fan-out runtime."""

    async def retrieve(variant: QueryVariant, candidate_k: int) -> Sequence[SearchHit]:
        return await asyncio.to_thread(
            engine.search,
            query_text=variant.query_text,
            query_embedding=variant.query_embedding,
            k=candidate_k,
            candidate_k=candidate_k,
        )

    return retrieve
