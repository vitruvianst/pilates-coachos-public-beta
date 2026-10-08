from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException
from firebase_admin import firestore

logger = logging.getLogger("ai_guard")
TAIPEI_TZ = timezone(timedelta(hours=8))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        logger.warning("Invalid integer env %s=%r; using %s", name, raw, default)
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        logger.warning("Invalid float env %s=%r; using %s", name, raw, default)
        return float(default)


AI_ENABLED = _env_bool("AI_ENABLED", True)
AI_GUARD_ENABLED = _env_bool("AI_GUARD_ENABLED", True)
AI_GUARD_FAIL_OPEN = _env_bool("AI_GUARD_FAIL_OPEN", False)
AI_GUARD_COLLECTION = os.environ.get("AI_GUARD_COLLECTION", "AIGuardUsage").strip() or "AIGuardUsage"
AI_GUARD_TTL_DAYS = _env_int("AI_GUARD_TTL_DAYS", 45)

# Public-beta default request ceilings. 0 means that particular ceiling is disabled.
AI_GLOBAL_DAILY_LIMIT = _env_int("AI_GLOBAL_DAILY_LIMIT", 3000)
AI_COMPANY_DAILY_LIMIT = _env_int("AI_COMPANY_DAILY_LIMIT", 500)
AI_COACH_DAILY_LIMIT = _env_int("AI_COACH_DAILY_LIMIT", 100)
AI_COACH_HOURLY_LIMIT = _env_int("AI_COACH_HOURLY_LIMIT", 30)
AI_COACH_MINUTE_LIMIT = _env_int("AI_COACH_MINUTE_LIMIT", 8)

# More expensive / abuse-prone operations get their own hourly ceiling.
AI_INBODY_HOURLY_LIMIT = _env_int("AI_INBODY_HOURLY_LIMIT", 10)
AI_QUESTIONNAIRE_HOURLY_LIMIT = _env_int("AI_QUESTIONNAIRE_HOURLY_LIMIT", 10)
AI_COMMUNICATION_HOURLY_LIMIT = _env_int("AI_COMMUNICATION_HOURLY_LIMIT", 30)

# Optional software-side cost breakers based on AIUsageMetrics pricing estimates.
# Keep 0 to disable until enough production observations are available.
AI_SOFT_GLOBAL_DAILY_USD = _env_float("AI_SOFT_GLOBAL_DAILY_USD", 0.0)
AI_SOFT_COMPANY_DAILY_USD = _env_float("AI_SOFT_COMPANY_DAILY_USD", 0.0)
AI_SOFT_COACH_DAILY_USD = _env_float("AI_SOFT_COACH_DAILY_USD", 0.0)


@dataclass(frozen=True)
class CounterSpec:
    key: str
    kind: str
    limit: int
    message: str
    company_id: str | None = None
    coach_id: str | None = None
    operation: str | None = None
    window: str | None = None


def _hash_key(*parts: str) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _operation_hourly_limit(operation: str) -> int:
    mapping = {
        "inbody_recognition": AI_INBODY_HOURLY_LIMIT,
        "questionnaire_recognition": AI_QUESTIONNAIRE_HOURLY_LIMIT,
        "coach_communication": AI_COMMUNICATION_HOURLY_LIMIT,
    }
    return int(mapping.get(operation, 0) or 0)


def _public_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _extract_identity(context: dict[str, Any] | None) -> tuple[str, str]:
    context = dict(context or {})
    company_id = str(context.get("company_id") or "").strip()
    coach_id = str(context.get("coach_id") or "").strip()
    if not company_id or not coach_id:
        raise _public_error(
            503,
            "AI 安全控管缺少 company_id / coach_id，已停止本次 AI 呼叫",
        )
    return company_id, coach_id


def _counter_specs(now: datetime, company_id: str, coach_id: str, operation: str) -> list[CounterSpec]:
    day = now.strftime("%Y%m%d")
    hour = now.strftime("%Y%m%d_%H")
    minute = now.strftime("%Y%m%d_%H%M")
    company_hash = _hash_key(company_id)
    coach_hash = _hash_key(company_id, coach_id)
    operation_hash = _hash_key(company_id, coach_id, operation)

    specs = [
        CounterSpec(
            key=f"day_{day}_global",
            kind="global_day",
            limit=AI_GLOBAL_DAILY_LIMIT,
            message="今日 AI 系統使用量已達安全上限，請稍後聯絡管理員",
            window=day,
        ),
        CounterSpec(
            key=f"day_{day}_company_{company_hash}",
            kind="company_day",
            limit=AI_COMPANY_DAILY_LIMIT,
            message="此場域今日 AI 使用量已達安全上限，請聯絡管理員",
            company_id=company_id,
            window=day,
        ),
        CounterSpec(
            key=f"day_{day}_coach_{coach_hash}",
            kind="coach_day",
            limit=AI_COACH_DAILY_LIMIT,
            message="此帳號今日 AI 使用量已達安全上限，請聯絡管理員",
            company_id=company_id,
            coach_id=coach_id,
            window=day,
        ),
        CounterSpec(
            key=f"hour_{hour}_coach_{coach_hash}",
            kind="coach_hour",
            limit=AI_COACH_HOURLY_LIMIT,
            message="AI 操作過於頻繁，請稍後再試",
            company_id=company_id,
            coach_id=coach_id,
            window=hour,
        ),
        CounterSpec(
            key=f"minute_{minute}_coach_{coach_hash}",
            kind="coach_minute",
            limit=AI_COACH_MINUTE_LIMIT,
            message="AI 操作過於頻繁，請稍候再試",
            company_id=company_id,
            coach_id=coach_id,
            window=minute,
        ),
    ]

    op_limit = _operation_hourly_limit(operation)
    if op_limit > 0:
        specs.append(
            CounterSpec(
                key=f"hour_{hour}_operation_{operation_hash}",
                kind="coach_operation_hour",
                limit=op_limit,
                message="此 AI 功能操作過於頻繁，請稍後再試",
                company_id=company_id,
                coach_id=coach_id,
                operation=operation,
                window=hour,
            )
        )
    return [spec for spec in specs if spec.limit > 0]


def _cost_cap_for_kind(kind: str) -> float:
    return {
        "global_day": AI_SOFT_GLOBAL_DAILY_USD,
        "company_day": AI_SOFT_COMPANY_DAILY_USD,
        "coach_day": AI_SOFT_COACH_DAILY_USD,
    }.get(kind, 0.0)


def require_ai_enabled() -> None:
    if not AI_ENABLED:
        raise _public_error(503, "AI 功能目前由管理員暫停")


def reserve_ai_call(
    db,
    *,
    service: str,
    operation: str,
    context: dict[str, Any] | None,
    request_id: str | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    """Atomically reserve one *actual provider attempt* before calling Gemini.

    Every retry calls this function again, so a retry storm cannot bypass the limits.
    The guard is shared by scanner_api and coach_api through one Firestore collection.
    """
    require_ai_enabled()
    if not AI_GUARD_ENABLED:
        return {"guard_enabled": False, "reserved": False}
    if db is None:
        raise _public_error(503, "AI 安全控管服務尚未就緒")

    company_id, coach_id = _extract_identity(context)
    now = datetime.now(TAIPEI_TZ)
    expires_at = now + timedelta(days=max(2, AI_GUARD_TTL_DAYS))
    specs = _counter_specs(now, company_id, coach_id, operation)
    refs = [(spec, db.collection(AI_GUARD_COLLECTION).document(spec.key)) for spec in specs]
    transaction = db.transaction()

    @firestore.transactional
    def _reserve(transaction):
        # Firestore requires reads before writes inside a transaction.
        snapshots = [ref.get(transaction=transaction) for _, ref in refs]

        for snapshot, (spec, _ref) in zip(snapshots, refs):
            data = snapshot.to_dict() if snapshot.exists else {}
            current = int((data or {}).get("count") or 0)
            if current >= spec.limit:
                logger.warning(
                    "AI_GUARD_BLOCK request_id=%s service=%s operation=%s kind=%s count=%s limit=%s company=%s coach=%s",
                    request_id,
                    service,
                    operation,
                    spec.kind,
                    current,
                    spec.limit,
                    company_id,
                    coach_id,
                )
                raise _public_error(429, spec.message)

            cost_cap = _cost_cap_for_kind(spec.kind)
            if cost_cap > 0:
                current_cost = float((data or {}).get("estimated_cost_usd") or 0.0)
                if current_cost >= cost_cap:
                    logger.warning(
                        "AI_GUARD_COST_BLOCK request_id=%s service=%s operation=%s kind=%s cost=%.6f cap=%.6f company=%s coach=%s",
                        request_id,
                        service,
                        operation,
                        spec.kind,
                        current_cost,
                        cost_cap,
                        company_id,
                        coach_id,
                    )
                    raise _public_error(429, "AI 預估費用已達今日軟性上限，請聯絡管理員")

        for snapshot, (spec, ref) in zip(snapshots, refs):
            data = snapshot.to_dict() if snapshot.exists else {}
            current = int((data or {}).get("count") or 0)
            payload = {
                "schema_version": "AIGuardUsage_v1",
                "kind": spec.kind,
                "window": spec.window,
                "count": current + 1,
                "limit": spec.limit,
                "company_id": spec.company_id,
                "coach_id": spec.coach_id,
                "operation": spec.operation,
                "last_service": service,
                "last_operation": operation,
                "last_request_id": request_id,
                "last_attempt": int(attempt),
                "updated_at": firestore.SERVER_TIMESTAMP,
                "expires_at": expires_at,
            }
            if not snapshot.exists:
                payload["created_at"] = firestore.SERVER_TIMESTAMP
                payload["estimated_cost_usd"] = 0.0
            transaction.set(ref, payload, merge=True)

        return {
            "guard_enabled": True,
            "reserved": True,
            "company_id": company_id,
            "coach_id": coach_id,
            "operation": operation,
            "attempt": int(attempt),
        }

    try:
        return _reserve(transaction)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("AI_GUARD_ERROR service=%s operation=%s: %s", service, operation, exc)
        if AI_GUARD_FAIL_OPEN:
            logger.error("AI_GUARD_FAIL_OPEN=true; allowing provider call despite guard error")
            return {"guard_enabled": True, "reserved": False, "fail_open": True}
        raise _public_error(503, "AI 安全控管暫時無法確認額度，已停止本次 AI 呼叫") from exc


def record_ai_cost(
    db,
    *,
    estimated_cost_usd: float | int | None,
    context: dict[str, Any] | None,
    service: str,
    operation: str,
) -> None:
    """Best-effort update of software-side estimated daily spend counters.

    This happens after the provider response, so failure here must never change the
    user-visible AI response. Google Cloud Spend Cap remains the hard billing layer.
    """
    if not AI_GUARD_ENABLED or db is None:
        return
    try:
        cost = float(estimated_cost_usd or 0.0)
    except (TypeError, ValueError):
        return
    if cost <= 0:
        return

    try:
        company_id, coach_id = _extract_identity(context)
        now = datetime.now(TAIPEI_TZ)
        day = now.strftime("%Y%m%d")
        company_hash = _hash_key(company_id)
        coach_hash = _hash_key(company_id, coach_id)
        keys = [
            f"day_{day}_global",
            f"day_{day}_company_{company_hash}",
            f"day_{day}_coach_{coach_hash}",
        ]
        batch = db.batch()
        for key in keys:
            ref = db.collection(AI_GUARD_COLLECTION).document(key)
            batch.set(
                ref,
                {
                    "estimated_cost_usd": firestore.Increment(cost),
                    "last_cost_service": service,
                    "last_cost_operation": operation,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
        batch.commit()
    except Exception as exc:
        # The AI call already happened; cost telemetry must not break the product.
        logger.exception("AI_GUARD_COST_RECORD_FAILED service=%s operation=%s: %s", service, operation, exc)


def public_status() -> dict[str, Any]:
    return {
        "ai_enabled": AI_ENABLED,
        "guard_enabled": AI_GUARD_ENABLED,
        "fail_open": AI_GUARD_FAIL_OPEN,
    }


def settings_for_log() -> dict[str, Any]:
    return {
        "ai_enabled": AI_ENABLED,
        "guard_enabled": AI_GUARD_ENABLED,
        "fail_open": AI_GUARD_FAIL_OPEN,
        "collection": AI_GUARD_COLLECTION,
        "limits": {
            "global_daily": AI_GLOBAL_DAILY_LIMIT,
            "company_daily": AI_COMPANY_DAILY_LIMIT,
            "coach_daily": AI_COACH_DAILY_LIMIT,
            "coach_hourly": AI_COACH_HOURLY_LIMIT,
            "coach_minute": AI_COACH_MINUTE_LIMIT,
            "inbody_hourly": AI_INBODY_HOURLY_LIMIT,
            "questionnaire_hourly": AI_QUESTIONNAIRE_HOURLY_LIMIT,
            "communication_hourly": AI_COMMUNICATION_HOURLY_LIMIT,
        },
        "soft_cost_usd": {
            "global_daily": AI_SOFT_GLOBAL_DAILY_USD,
            "company_daily": AI_SOFT_COMPANY_DAILY_USD,
            "coach_daily": AI_SOFT_COACH_DAILY_USD,
        },
        "ttl_days": AI_GUARD_TTL_DAYS,
    }


def log_settings() -> None:
    logger.info("AI_GUARD_SETTINGS %s", json.dumps(settings_for_log(), ensure_ascii=False))
