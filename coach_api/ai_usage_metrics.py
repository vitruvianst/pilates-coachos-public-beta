from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from typing import Any

from firebase_admin import firestore

logger = logging.getLogger("ai_usage_metrics")
TAIPEI_TZ = timezone(timedelta(hours=8))
COLLECTION_NAME = os.environ.get("AI_USAGE_COLLECTION", "AIUsageMetrics").strip() or "AIUsageMetrics"
SCHEMA_VERSION = "AIUsageMetrics_v1"
PRICING_SOURCE_URL = "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing"

# Current CoachOS configuration: Vertex AI global + Priority PayGo.
# Store the applied rate snapshot on every record so historical estimates remain auditable
# after Google changes list prices.
_PRIORITY_GLOBAL_PRICING = {
    "gemini-3.5-flash": [
        {
            "effective_from": "2026-01-01",
            "effective_to": None,
            "pricing_version": "vertex_gemini_3_5_flash_priority_global",
            "input_usd_per_1m": 2.70,
            "cached_input_usd_per_1m": 0.27,
            "output_usd_per_1m": 16.20,
        },
    ],
}

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai-usage")


def new_request_id() -> str:
    return uuid.uuid4().hex


def pseudonymous_member_ref(company_id: str | None, branch_id: str | None, used_id: str | None) -> str | None:
    if not used_id:
        return None
    salt = os.environ.get("AI_USAGE_HASH_SALT", "coachos-ai-usage-v1")
    raw = "|".join([salt, str(company_id or ""), str(branch_id or ""), str(used_id or "")])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raw = getattr(value, "value", None)
    if raw is not None and isinstance(raw, (str, int, float, bool)):
        return raw
    return str(value)


def _int_attr(obj: Any, name: str) -> int:
    try:
        value = getattr(obj, name, 0) if obj is not None else 0
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _modality_map(details: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in details or []:
        modality = _scalar(getattr(item, "modality", None)) or "UNKNOWN"
        token_count = _int_attr(item, "token_count")
        key = str(modality).replace("Modality.", "").upper()
        out[key] = out.get(key, 0) + token_count
    return out


def extract_usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "thought_tokens": 0,
            "cached_tokens": 0,
            "tool_use_tokens": 0,
            "total_tokens": 0,
            "traffic_type": None,
            "modality_tokens": {},
        }

    return {
        "prompt_tokens": _int_attr(usage, "prompt_token_count"),
        "candidate_tokens": _int_attr(usage, "candidates_token_count"),
        "thought_tokens": _int_attr(usage, "thoughts_token_count"),
        "cached_tokens": _int_attr(usage, "cached_content_token_count"),
        "tool_use_tokens": _int_attr(usage, "tool_use_prompt_token_count"),
        "total_tokens": _int_attr(usage, "total_token_count"),
        "traffic_type": _scalar(getattr(usage, "traffic_type", None)),
        "modality_tokens": {
            "prompt": _modality_map(getattr(usage, "prompt_tokens_details", None)),
            "candidate": _modality_map(getattr(usage, "candidates_tokens_details", None)),
            "cache": _modality_map(getattr(usage, "cache_tokens_details", None)),
            "tool_use_prompt": _modality_map(getattr(usage, "tool_use_prompt_tokens_details", None)),
        },
    }


def resolve_pricing(model: str, at: datetime | None = None) -> dict[str, Any] | None:
    at = at or datetime.now(TAIPEI_TZ)
    day = at.date().isoformat()
    rows = _PRIORITY_GLOBAL_PRICING.get(model, [])
    for row in rows:
        if day < row["effective_from"]:
            continue
        if row["effective_to"] and day > row["effective_to"]:
            continue
        return dict(row)
    return None


def estimate_cost_usd(usage: dict[str, Any], pricing: dict[str, Any] | None) -> dict[str, Any]:
    if not pricing:
        return {
            "status": "pricing_unavailable",
            "input_usd": None,
            "cached_input_usd": None,
            "output_usd": None,
            "estimated_cost_usd": None,
        }

    prompt = max(0, int(usage.get("prompt_tokens") or 0))
    cached = max(0, min(prompt, int(usage.get("cached_tokens") or 0)))
    uncached = max(0, prompt - cached)
    output = max(0, int(usage.get("candidate_tokens") or 0)) + max(0, int(usage.get("thought_tokens") or 0))

    input_cost = uncached / 1_000_000 * float(pricing["input_usd_per_1m"])
    cached_cost = cached / 1_000_000 * float(pricing["cached_input_usd_per_1m"])
    output_cost = output / 1_000_000 * float(pricing["output_usd_per_1m"])
    total = input_cost + cached_cost + output_cost

    return {
        "status": "estimated_from_usage_metadata",
        "billable_uncached_input_tokens": uncached,
        "billable_cached_input_tokens": cached,
        "billable_output_tokens": output,
        "input_usd": round(input_cost, 10),
        "cached_input_usd": round(cached_cost, 10),
        "output_usd": round(output_cost, 10),
        "estimated_cost_usd": round(total, 10),
    }


def build_metric_record(
    *,
    request_id: str,
    service: str,
    operation: str,
    model: str,
    location: str,
    requested_mode: str,
    response: Any | None,
    total_latency_ms: int,
    model_latency_ms: int | None,
    attempt_count: int,
    success: bool,
    error_status: int | str | None = None,
    error_message: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(TAIPEI_TZ)
    context = dict(context or {})
    usage = extract_usage_metadata(response)
    pricing = resolve_pricing(model, now) if success else None
    cost = estimate_cost_usd(usage, pricing) if success else {
        "status": "not_estimated_failed_request",
        "input_usd": None,
        "cached_input_usd": None,
        "output_usd": None,
        "estimated_cost_usd": None,
    }

    company_id = context.get("company_id")
    branch_id = context.get("branch_id")
    coach_id = context.get("coach_id")
    used_id = context.get("used_id") or context.get("target_user_id")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "created_at": firestore.SERVER_TIMESTAMP,
        "created_at_local": now.strftime("%Y-%m-%d %H:%M:%S"),
        "record_date": now.strftime("%Y-%m-%d"),
        "environment": os.environ.get("COACHOS_ENV", "public_beta"),
        "service": service,
        "operation": operation,
        "provider": "vertex_ai",
        "model": model,
        "location": location,
        "requested_mode": requested_mode,
        "reported_traffic_type": usage.get("traffic_type"),
        "prompt_version": context.get("prompt_version"),
        "schema_output_version": context.get("schema_output_version"),
        "company_id": company_id,
        "branch_id": branch_id,
        "coach_id": coach_id,
        "member_ref_hash": pseudonymous_member_ref(company_id, branch_id, used_id),
        "input_context": {
            "image_count": int(context.get("image_count") or 0),
            "image_bytes": int(context.get("image_bytes") or 0),
        },
        "tokens": usage,
        "performance": {
            "model_latency_ms": model_latency_ms,
            "total_latency_ms": int(total_latency_ms),
            "attempt_count": int(attempt_count),
            "retry_count": max(0, int(attempt_count) - 1),
        },
        "result": {
            "model_call_success": bool(success),
            "error_status": _scalar(error_status),
            "error_message": (str(error_message)[:500] if error_message else None),
        },
        "pricing": {
            "source": PRICING_SOURCE_URL,
            "currency": "USD",
            **(pricing or {}),
        },
        "cost": cost,
    }
    return record


def _write_metric(db, request_id: str, record: dict[str, Any]) -> None:
    db.collection(COLLECTION_NAME).document(request_id).set(record, merge=True)


def save_metric_async(db, record: dict[str, Any]) -> None:
    """Best-effort telemetry write. Never blocks or breaks the product request."""
    request_id = str(record.get("request_id") or new_request_id())
    record["request_id"] = request_id

    # Structured log is a second evidence trail if Firestore telemetry has a transient issue.
    try:
        logger.info("AI_USAGE_METRIC %s", json.dumps({
            "request_id": request_id,
            "service": record.get("service"),
            "operation": record.get("operation"),
            "model": record.get("model"),
            "tokens": record.get("tokens"),
            "cost": record.get("cost"),
            "performance": record.get("performance"),
            "result": record.get("result"),
        }, ensure_ascii=False, default=str))
    except Exception:
        logger.exception("AI usage structured log failed")

    def task():
        last_exc = None
        for attempt in range(2):
            try:
                _write_metric(db, request_id, record)
                return
            except Exception as exc:  # telemetry failure must never fail CoachOS
                last_exc = exc
                if attempt == 0:
                    time.sleep(0.5)
        logger.error("AIUsageMetrics write failed request_id=%s error=%s", request_id, last_exc)

    try:
        _EXECUTOR.submit(task)
    except Exception:
        logger.exception("AIUsageMetrics enqueue failed request_id=%s", request_id)
