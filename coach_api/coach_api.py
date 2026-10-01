import os
import json
import logging
import re
import random
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
from google import genai
from google.genai import types

from assessment_specs import ASSESSMENT_SPECS
from gait_norms import build_gait_references
from ai_usage_metrics import build_metric_record, new_request_id, save_metric_async


load_dotenv()

# =============================================================================
# CoachOS Coach API
# - Read-only toward existing HealthRecords except today_status
# - Deterministic rule engine for exercise selection
# - Gemini is used ONLY for communication wording/tone
# =============================================================================

TAIPEI_TZ = timezone(timedelta(hours=8))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FIREBASE_PROJECT_ID = os.environ.get(
    "FIREBASE_PROJECT_ID",
    "",
).strip()
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", FIREBASE_PROJECT_ID).strip()
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "firebase_key.json").strip()

BASE_DIR = Path(__file__).resolve().parent
MAIN_RULE_DIR = Path(os.environ.get("MAIN_RULE_DIR", BASE_DIR / "main_rule"))
POSE_CATALOG_PATH = Path(os.environ.get("POSE_CATALOG_PATH", BASE_DIR / "pose_catalog.json"))
ISSUE_COUNT_PATH = Path(os.environ.get("ISSUE_COUNT_PATH", BASE_DIR / "issue_count.json"))
HISTORY_BLOCK_PATH = Path(os.environ.get("HISTORY_BLOCK_PATH", BASE_DIR / "history_block.json"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("coach_api")

app = FastAPI(
    title="CoachOS Coach API",
    version="10.0.0",
    description="CoachOS v10：Firebase nested-field 防 crash、三態資料完整度、部分資料容錯與跨教練歷史 API",
)

COACH_WEB_ORIGINS = [
    x.strip() for x in os.environ.get(
        "COACH_WEB_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=COACH_WEB_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)


def _resolve_service_account_path() -> str | None:
    raw = Path(FIREBASE_KEY_PATH)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(BASE_DIR / raw)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _project_id_from_service_account(key_path: str | None) -> str:
    if not key_path:
        return ""
    try:
        with open(key_path, "r", encoding="utf-8") as fh:
            return str((json.load(fh) or {}).get("project_id") or "").strip()
    except Exception:
        return ""


def _build_vertex_client(project_id: str):
    if not project_id:
        raise RuntimeError(
            "Vertex AI 缺少 GOOGLE_CLOUD_PROJECT / FIREBASE_PROJECT_ID"
        )

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=GOOGLE_CLOUD_LOCATION,
        http_options=types.HttpOptions(
            api_version="v1",
            headers={
                "X-Vertex-AI-LLM-Request-Type": "shared",
                "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
            },
        ),
    )


VERTEX_RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

VERTEX_RETRY_BASE_DELAYS = (
    2.0,
    5.0,
    10.0,
    20.0,
)


def _vertex_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)

        try:
            if value is not None:
                code = int(value)
                if 100 <= code <= 599:
                    return code
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)

    value = (
        getattr(response, "status_code", None)
        if response is not None
        else None
    )

    try:
        if value is not None:
            code = int(value)
            if 100 <= code <= 599:
                return code
    except (TypeError, ValueError):
        pass

    message = str(exc or "")

    for code in VERTEX_RETRYABLE_STATUS_CODES:
        if (
            message.startswith(str(code))
            or f"'code': {code}" in message
            or f'"code": {code}' in message
        ):
            return code

    return None


def _generate_content_with_retry(
    client,
    *,
    model: str,
    contents,
    config,
    operation: str,
    metric_context: dict[str, Any] | None = None,
):
    """Call Vertex Gemini with bounded retry and emit non-blocking AIUsageMetrics.

    Telemetry is sidecar-only. Failure to write AIUsageMetrics never changes the
    Coach API response or communication fallback behavior.
    """
    total_attempts = 1 + len(VERTEX_RETRY_BASE_DELAYS)
    last_error = None
    request_id = new_request_id()
    logical_started = time.perf_counter()

    for attempt in range(1, total_attempts + 1):
        attempt_started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            model_latency_ms = round((time.perf_counter() - attempt_started) * 1000)
            total_latency_ms = round((time.perf_counter() - logical_started) * 1000)

            usage = getattr(response, "usage_metadata", None)
            traffic_type = getattr(usage, "traffic_type", None)
            logger.info(
                "Vertex AI %s success: traffic_type=%s",
                operation,
                traffic_type or "unknown",
            )

            db = getattr(app.state, "db", None)
            if db is not None:
                record = build_metric_record(
                    request_id=request_id,
                    service="coach_api",
                    operation=operation,
                    model=model,
                    location=GOOGLE_CLOUD_LOCATION,
                    requested_mode="priority",
                    response=response,
                    total_latency_ms=total_latency_ms,
                    model_latency_ms=model_latency_ms,
                    attempt_count=attempt,
                    success=True,
                    context=metric_context,
                )
                save_metric_async(db, record)

            return response

        except Exception as exc:
            last_error = exc
            status = _vertex_status_code(exc)
            retryable = status in VERTEX_RETRYABLE_STATUS_CODES

            if not retryable or attempt >= total_attempts:
                total_latency_ms = round((time.perf_counter() - logical_started) * 1000)
                db = getattr(app.state, "db", None)
                if db is not None:
                    record = build_metric_record(
                        request_id=request_id,
                        service="coach_api",
                        operation=operation,
                        model=model,
                        location=GOOGLE_CLOUD_LOCATION,
                        requested_mode="priority",
                        response=None,
                        total_latency_ms=total_latency_ms,
                        model_latency_ms=None,
                        attempt_count=attempt,
                        success=False,
                        error_status=status,
                        error_message=str(exc),
                        context=metric_context,
                    )
                    save_metric_async(db, record)

                logger.error(
                    "Vertex AI %s failed: status=%s attempt=%s/%s error=%s",
                    operation, status, attempt, total_attempts, exc,
                )
                raise

            base = VERTEX_RETRY_BASE_DELAYS[attempt - 1]
            wait_seconds = round(base * random.uniform(0.8, 1.2), 2)
            logger.warning(
                "Vertex AI %s transient error: status=%s attempt=%s/%s; retry in %.2fs",
                operation, status, attempt, total_attempts, wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Vertex AI {operation} failed: {last_error}")


@app.on_event("startup")
def startup_event():
    key_path = _resolve_service_account_path()

    project_id = (
        GOOGLE_CLOUD_PROJECT
        or FIREBASE_PROJECT_ID
        or _project_id_from_service_account(key_path)
    )

    if key_path:
        # 同一 service account 同時給
        # Firebase Admin + Vertex AI ADC 使用。
        os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"
        ] = key_path

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                credentials.Certificate(key_path),
                options={
                    "projectId": project_id
                } if project_id else None,
            )

        logger.info(
            "✅ Coach API 使用後端 service account：%s",
            key_path,
        )

    else:
        if not firebase_admin._apps:
            if not project_id:
                raise RuntimeError(
                    "Firebase / Vertex AI 缺少 project ID"
                )

            firebase_admin.initialize_app(
                options={
                    "projectId": project_id
                }
            )

    app.state.db = firestore.client()
    app.state.google_cloud_project = project_id

    app.state.genai_client = (
        _build_vertex_client(project_id)
    )

    logger.info(
        "🚀 Coach API Firebase + Vertex AI "
        "Priority PayGo 已就緒 "
        "project=%s location=%s model=%s",
        project_id,
        GOOGLE_CLOUD_LOCATION,
        GEMINI_MODEL,
    )


# =============================================================================
# Canonical field definitions
# =============================================================================

BODY_STATUS_KEYS = [
    "discomfort_neck",
    "discomfort_shoulder",
    "discomfort_elbow",
    "discomfort_wrist_hand",
    "discomfort_upper_back",
    "discomfort_low_back",
    "discomfort_hip",
    "discomfort_knee",
    "discomfort_ankle_foot",
]

BODY_STATUS_LABELS = {
    "discomfort_neck": "頸部",
    "discomfort_shoulder": "肩部",
    "discomfort_elbow": "手肘",
    "discomfort_wrist_hand": "手腕／手部",
    "discomfort_upper_back": "上背",
    "discomfort_low_back": "下背",
    "discomfort_hip": "髖部",
    "discomfort_knee": "膝部",
    "discomfort_ankle_foot": "踝部／足部",
}

QUESTIONNAIRE_LABELS = {
    # intensity
    "prefers_light_intensity": "輕鬆強度",
    "prefers_moderate_intensity": "中等強度",
    "prefers_vigorous_intensity": "較有挑戰的高強度",
    # exercise preferences
    "likes_walking": "散步",
    "likes_brisk_walking": "健走",
    "likes_jogging": "慢跑",
    "likes_running": "跑步",
    "likes_marathon_or_road_race": "馬拉松／路跑",
    "likes_hiking": "登山",
    "likes_trail_activity": "步道／戶外越野",
    "likes_cycling": "自行車",
    "likes_aquatic_exercise": "水中運動",
    "likes_resistance_training": "重量／阻力訓練",
    "likes_yoga": "瑜伽",
    "likes_pilates": "皮拉提斯",
    "likes_tai_chi_or_qigong": "太極／氣功",
    "likes_dance": "舞蹈",
    "likes_racket_sports": "拍類運動",
    "likes_ball_sports": "球類運動",
    "likes_golf": "高爾夫",
    "likes_outdoor_fitness": "戶外運動／戶外健身",
    # social
    "prefers_exercise_alone": "自己運動",
    "prefers_one_on_one_coaching": "一對一陪同",
    "prefers_small_group": "小團體",
    "prefers_large_group_class": "多人團課",
    "prefers_with_family_or_friends": "與家人朋友一起活動",
    # goals
    "goal_general_health": "維持／提升整體健康",
    "goal_weight_management": "控制體重／體脂",
    "goal_body_shape": "改善體態",
    "goal_strength_or_muscle_gain": "增加肌力或肌肉量",
    "goal_mobility": "提升柔軟度／關節活動度",
    "goal_balance": "提升平衡",
    "goal_fall_prevention": "降低跌倒風險",
    "goal_independence": "維持獨立生活能力",
    "goal_travel_fitness": "為旅遊／走路／行程體力做準備",
    "goal_sleep_stress": "改善睡眠／壓力",
    "goal_social_connection": "增加社交",
    "goal_build_exercise_habit": "建立規律運動習慣",
    "goal_energy_vitality": "更有精神與活力",
    # communication
    "communication_prefers_step_by_step": "一步一步清楚說明",
    "communication_prefers_demonstration": "先看示範再做",
    "communication_prefers_data_progress": "看數據與進步幅度",
    "communication_prefers_encouragement": "鼓勵與正向回饋",
    "communication_prefers_challenge": "有挑戰與目標感",
    "communication_prefers_casual_chat": "輕鬆聊天式互動",
    "communication_prefers_quiet_focus": "安靜專注、少聊天",
    "communication_prefers_variety": "每次有些變化",
    "communication_prefers_routine": "固定熟悉的流程",
    "communication_prefers_slow_pace_explanation": "較慢速度、重點清楚的說明",
    # travel
    "likes_travel": "喜歡旅遊",
    "travel_frequency_none": "沒有旅遊",
    "travel_frequency_1_per_year": "約每年 1 次旅遊",
    "travel_frequency_2_3_per_year": "每年旅遊 2–3 次",
    "travel_frequency_4_plus_per_year": "每年旅遊 4 次以上",
    "travel_scope_domestic_more": "國內旅遊為主",
    "travel_scope_international_more": "國外旅遊為主",
    "travel_scope_both": "國內與國外都常安排",
    "travel_destination_domestic": "台灣國內",
    "travel_destination_japan": "日本",
    "travel_destination_korea": "韓國",
    "travel_destination_southeast_asia": "東南亞",
    "travel_destination_greater_china_hk_macau": "中國大陸／香港／澳門",
    "travel_destination_europe": "歐洲",
    "travel_destination_north_america": "北美",
    "travel_destination_oceania": "澳洲／紐西蘭",
    "travel_destination_other_international": "其他國外地區",
    "travel_style_independent": "自由行",
    "travel_style_group_tour": "跟團",
    "travel_style_cruise": "郵輪",
    "travel_style_road_trip": "自駕／公路旅行",
    "travel_style_active": "活動型旅遊",
    "travel_style_food_culture": "美食／文化／城市型旅遊",
    "travel_style_nature": "自然景觀型旅遊",
    # leisure / self description / information style
    "leisure_outdoor_activity": "戶外活動",
    "leisure_hiking_nature": "山林／自然景點",
    "leisure_gardening": "園藝",
    "leisure_photography": "攝影",
    "leisure_reading": "閱讀",
    "self_description_routine_oriented": "喜歡固定熟悉的節奏",
    "self_description_adventurous": "喜歡冒險／探索",
    "self_description_patient": "有耐心",
    "comfortable_using_smartphone": "使用智慧型手機很自在",
    "likes_app_progress_tracking": "喜歡用 App 看進度／紀錄",
    "prefers_printed_information": "偏好紙本資訊",
    "prefers_large_text": "偏好較大文字與清楚版面",
}


# =============================================================================
# Models
# =============================================================================

class TodayStatusRequest(BaseModel):
    branch_id: str
    used_id: str
    user_name: str | None = None
    intensity_status: Literal["light", "moderate", "vigorous"]
    body_status: dict[str, bool] = Field(default_factory=dict)


class CommunicationOutput(BaseModel):
    communication_advice: str
    coach_script: str
    tone_tags: list[str]
    tone_guidance: str


class Actor(BaseModel):
    company_id: str
    branch_id: str | None = None
    coach_id: str
    email: str
    display_name: str | None = None
    role: Literal["coach", "manager", "owner"]
    firebase_uid: str


# =============================================================================
# Generic helpers
# =============================================================================

def _db():
    if not hasattr(app.state, "db"):
        raise HTTPException(status_code=503, detail="Firebase 尚未完成初始化")
    return app.state.db


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_dict(value: Any) -> dict:
    """Return value only when it is a dict; malformed/missing nested Firebase data becomes {}."""
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list:
    """Return value only when it is a list; malformed/missing nested Firebase data becomes []."""
    return value if isinstance(value, list) else []


_MISSING = object()


def _path_exists(data: dict | None, path: str) -> bool:
    return _deep_get(data, path, _MISSING) is not _MISSING


def _has_scalar_value(data: dict | None, path: str) -> bool:
    value = _deep_get(data, path, _MISSING)
    return value is not _MISSING and value is not None and value != ""


def _has_dict_section(data: dict | None, path: str, allow_empty: bool = False) -> bool:
    value = _deep_get(data, path, _MISSING)
    if not isinstance(value, dict):
        return False
    return allow_empty or bool(value)


def _sanitize_doc_id_part(value: Any, fallback: str) -> str:
    text = _clean(value) or fallback
    return text.replace("/", "-").replace("\\", "-")


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _deep_get(data: dict | None, path: str, default=None):
    cur: Any = data or {}
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _unique(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        text = _clean(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=TAIPEI_TZ)
    if hasattr(value, "to_datetime"):
        try:
            dt = value.to_datetime()
            return dt if dt.tzinfo else dt.replace(tzinfo=TAIPEI_TZ)
        except Exception:
            pass
    text = _clean(value)
    if not text:
        return None
    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d_%H%M%S",
        "%Y%m%d",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=TAIPEI_TZ)
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=TAIPEI_TZ)
    except Exception:
        return None


def _record_sort_time(record: dict) -> datetime:
    """
    HealthRecords 的時間排序以「實際資料發生時間」為準，
    不以資料何時上傳、同步或修改來判斷新舊。

    Priority:
    1. recorded_at       : 實際事件/量測時間（canonical timestamp）
    2. recorded_at_local : 實際事件/量測的本地時間
    3. record_date       : 實際事件/量測日期

    Legacy fallback only:
    4. createdAt             : Firestore 首次建立時間
    5. sync_at               : 同步時間
    6. updatedAt             : 更新時間
    7. last_updated_at_local : 最後內容修改時間

    因此舊體測即使今天才補上傳，也不會被判定為今天的新體測。
    """

    # ---------------------------------------------------------
    # A. Business / assessment time
    # ---------------------------------------------------------
    for key in [
        "recorded_at",
        "recorded_at_local",
        "record_date",
    ]:
        dt = _parse_datetime(record.get(key))
        if dt is not None:
            return dt

    # ---------------------------------------------------------
    # B. Legacy fallback
    # 僅限沒有任何 assessment/event time 的舊資料
    # ---------------------------------------------------------
    for key in [
        "createdAt",
        "sync_at",
        "updatedAt",
        "last_updated_at_local",
    ]:
        dt = _parse_datetime(record.get(key))
        if dt is not None:
            return dt

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _record_display_date(record: dict | None) -> str | None:
    if not record:
        return None
    if record.get("record_date"):
        return str(record["record_date"])
    dt = _record_sort_time(record)
    if dt.year > 1970:
        return dt.astimezone(TAIPEI_TZ).strftime("%Y-%m-%d")
    return None


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Rule Base 檔案不存在：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("root must be object")
        return data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"無法讀取 JSON：{path.name}: {exc}")


# =============================================================================
# Authentication, authorization, and Firestore readers
# =============================================================================

def _query_rows(collection_name: str, filters: list[tuple[str, str, Any]]) -> list[dict]:
    query = _db().collection(collection_name)
    for field, op, value in filters:
        query = query.where(field, op, value)
    rows: list[dict] = []
    for snap in query.stream():
        row = snap.to_dict() or {}
        row["_doc_id"] = snap.id
        rows.append(row)
    return rows


def _resolve_actor(decoded: dict) -> Actor:
    """
    Firebase ID token 只證明身分；
    Firestore Coaches/{uid} 才是 CoachOS 權限來源。
    """
    email = _clean(
        decoded.get("email")
    ).lower()

    uid = _clean(
        decoded.get("uid")
        or decoded.get("user_id")
        or decoded.get("sub")
    )

    if not uid:
        raise HTTPException(
            status_code=401,
            detail="Firebase token 缺少 UID",
        )

    # Canonical Public Beta path
    snap = (
        _db()
        .collection("Coaches")
        .document(uid)
        .get()
    )

    if snap.exists:
        data = snap.to_dict() or {}

    else:
        # Legacy fallback：
        # 舊 Coaches 文件若不是 UID document ID，
        # 只允許唯一 email 命中。
        candidates = []

        if email:
            candidates = _query_rows(
                "Coaches",
                [
                    ("email", "==", email),
                ],
            )

        if len(candidates) != 1:
            raise HTTPException(
                status_code=403,
                detail=(
                    "此 Firebase 帳號尚未建立唯一的 "
                    "Coaches 權限資料"
                ),
            )

        data = candidates[0]

    if data.get("disabled") is True:
        raise HTTPException(
            status_code=403,
            detail="此帳號已停用",
        )

    role = _clean(
        data.get("role")
    ).lower()

    if role not in {
        "coach",
        "manager",
        "owner",
    }:
        raise HTTPException(
            status_code=403,
            detail=(
                f"不支援的 role："
                f"{role or '(empty)'}"
            ),
        )

    company_id = _clean(
        data.get("company_id")
    )

    branch_id = (
        _clean(data.get("branch_id"))
        or None
    )

    coach_id = _clean(
        data.get("coach_id")
    )

    if not company_id or not coach_id:
        raise HTTPException(
            status_code=403,
            detail=(
                "Coaches 缺少 "
                "company_id 或 coach_id"
            ),
        )

    if (
        role in {"coach", "manager"}
        and not branch_id
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                f"role={role} "
                "必須設定 branch_id"
            ),
        )

    return Actor(
        company_id=company_id,
        branch_id=branch_id,
        coach_id=coach_id,
        email=(
            email
            or _clean(data.get("email"))
        ),
        display_name=(
            _clean(
                data.get("display_name")
            )
            or None
        ),
        role=role,
        firebase_uid=uid,
    )


def require_actor(authorization: str | None = Header(default=None)) -> Actor:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="缺少 Firebase Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Firebase token 為空")
    try:
        decoded = firebase_auth.verify_id_token(token, clock_skew_seconds=5)
    except Exception as exc:
        logger.warning("Firebase token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="登入已失效，請重新登入") from exc
    return _resolve_actor(decoded)


def _enforce_branch(actor: Actor, requested_branch_id: str) -> str:
    branch_id = _clean(requested_branch_id)
    if not branch_id:
        raise HTTPException(status_code=422, detail="branch_id 不可為空")
    if actor.role in {"coach", "manager"} and branch_id != actor.branch_id:
        raise HTTPException(status_code=403, detail="無權存取其他分店")
    return branch_id


def _association_exists(actor: Actor, branch_id: str, used_id: str) -> bool:
    """Coach visibility is granted only by Students association.

    HealthRecords records who uploaded/performed the assessment, but does not
    itself grant member-read permission. This keeps upload identity and read
    authorization separated when manager/owner also have coach capability.
    """
    filters = [
        ("company_id", "==", actor.company_id),
        ("branch_id", "==", branch_id),
        ("coach_id", "==", actor.coach_id),
        ("used_id", "==", used_id),
    ]
    query = _db().collection("Students")
    for field, op, value in filters:
        query = query.where(field, op, value)
    return next(iter(query.limit(1).stream()), None) is not None


def _assert_member_access(actor: Actor, branch_id: str, used_id: str) -> None:
    branch_id = _enforce_branch(actor, branch_id)
    used_id = _clean(used_id)
    if not used_id:
        raise HTTPException(status_code=422, detail="used_id 不可為空")

    if actor.role == "owner":
        return
    if actor.role == "manager":
        return
    if actor.role == "coach" and _association_exists(actor, branch_id, used_id):
        return
    raise HTTPException(status_code=403, detail="此學員不在目前教練的可見範圍")


def _fetch_member_health_records(actor: Actor, branch_id: str, used_id: str) -> list[dict]:
    """After access is granted, compare ALL coaches' records for the same member."""
    branch_id = _enforce_branch(actor, branch_id)
    used_id = _clean(used_id)
    _assert_member_access(actor, branch_id, used_id)
    records = _query_rows(
        "HealthRecords",
        [
            ("company_id", "==", actor.company_id),
            ("branch_id", "==", branch_id),
            ("used_id", "==", used_id),
        ],
    )
    records.sort(key=_record_sort_time, reverse=True)
    return records


def _by_type(records: list[dict], record_type: str) -> list[dict]:
    rows = [r for r in records if _clean(r.get("type")) == record_type]
    rows.sort(key=_record_sort_time, reverse=True)
    return rows


def _latest(records: list[dict], record_type: str) -> dict | None:
    rows = _by_type(records, record_type)
    return rows[0] if rows else None


def _current_today_status(records: list[dict]) -> dict | None:
    latest = _latest(records, "today_status")
    if not latest:
        return None
    today = datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d")
    return latest if _record_display_date(latest) == today else None


def _recent_with_section(records: list[dict], section_path: str, limit: int = 2) -> list[dict]:
    """Return the latest biomechanics records from DISTINCT test dates.

    HealthRecords may contain more than one biomechanics document on the same
    calendar day (for example a re-upload or a substitute coach upload).  The
    CoachOS comparison UI is intended to compare visits/test days, not duplicate
    documents from one day.

    `_by_type()` is already sorted newest -> oldest, therefore the first record
    encountered for a date is the newest record for that test date.
    """
    out: list[dict] = []
    used_dates: set[str] = set()

    for row in _by_type(records, "biomechanics"):
        section = _deep_get(row, section_path)
        if not isinstance(section, dict) or not section:
            continue

        test_date = _record_display_date(row)
        if not test_date:
            # A comparison column must have a real test date.  Do not let an
            # undated document displace a valid dated visit.
            continue

        if test_date in used_dates:
            continue

        used_dates.add(test_date)
        out.append(row)

        if len(out) >= limit:
            break

    return out


def _member_scope_rows(actor: Actor) -> tuple[list[dict], list[dict]]:
    """Return candidate Students and HealthRecords according to role scope."""
    if actor.role == "coach":
        common = [
            ("company_id", "==", actor.company_id),
            ("branch_id", "==", actor.branch_id),
            ("coach_id", "==", actor.coach_id),
        ]
    elif actor.role == "manager":
        common = [
            ("company_id", "==", actor.company_id),
            ("branch_id", "==", actor.branch_id),
        ]
    else:  # owner
        common = [("company_id", "==", actor.company_id)]

    return _query_rows("Students", common), _query_rows("HealthRecords", common)


def _visible_members(actor: Actor) -> list[dict]:
    student_rows, health_rows = _member_scope_rows(actor)
    members: dict[tuple[str, str], dict] = {}

    def upsert(data: dict, source: str):
        branch_id = _clean(data.get("branch_id"))
        used_id = _clean(data.get("used_id"))
        if not branch_id or not used_id:
            return
        key = (branch_id, used_id)
        item = members.setdefault(
            key,
            {
                "used_id": used_id,
                "user_name": used_id,
                "company_id": actor.company_id,
                "branch_id": branch_id,
                "known_names": set(),
                "has_health_records": False,
                "_latest_name_time": datetime(1970, 1, 1, tzinfo=timezone.utc),
            },
        )
        name = _clean(data.get("user_name") or data.get("name"))
        if name:
            item["known_names"].add(name)
            ts = _record_sort_time(data)
            if ts >= item["_latest_name_time"]:
                item["user_name"] = name
                item["_latest_name_time"] = ts
        if source == "health":
            item["has_health_records"] = True

    for row in student_rows:
        upsert(row, "student")
    for row in health_rows:
        upsert(row, "health")

    output = []
    for item in members.values():
        item.pop("_latest_name_time", None)
        item["known_names"] = sorted(item["known_names"])
        item["name_conflict"] = len(item["known_names"]) > 1
        output.append(item)

    output.sort(key=lambda x: (x["branch_id"], x["user_name"], x["used_id"]))
    return output

# =============================================================================
# Rule-key normalization
# =============================================================================

def _normalize_gender(value: Any) -> str | None:
    text = _clean(value).lower()
    if text in {"m", "male", "man", "男性", "男"}:
        return "male"
    if text in {"f", "female", "woman", "女性", "女"}:
        return "female"
    return None


def _age_group(age: int) -> str:
    if 18 <= age <= 29:
        return "youth"
    if 30 <= age <= 49:
        return "adult"
    if 50 <= age <= 64:
        return "mid"
    if age >= 65:
        return "senior"
    # Rule base defined from 18+. Under-18 should not silently map into an adult rule.
    raise HTTPException(status_code=422, detail=f"age={age} 不在目前 Rule Base 支援範圍（18+）")


def _resolve_profile(bodycomp: dict | None, biomechanics: dict | None) -> dict:
    bodycomp = bodycomp or {}
    biomechanics = biomechanics or {}

    bio_age = _to_int(_deep_get(biomechanics, "profile.age"))
    body_age = _to_int(bodycomp.get("age"))
    if body_age is None:
        body_age = _to_int(_deep_get(bodycomp, "profile.age"))

    bio_gender = _normalize_gender(_deep_get(biomechanics, "profile.gender"))
    body_gender = _normalize_gender(bodycomp.get("gender"))

    age = bio_age if bio_age is not None else body_age
    gender = bio_gender or body_gender

    if age is None:
        raise HTTPException(status_code=422, detail="Age 缺失：體測與 Body Composition 都沒有可用年齡")
    if gender is None:
        raise HTTPException(status_code=422, detail="Gender 缺失：體測與 Body Composition 都沒有可用性別")

    return {
        "age": age,
        "age_group": _age_group(age),
        "gender": gender,
        "age_source": "biomechanics" if bio_age is not None else "body_composition",
        "gender_source": "biomechanics" if bio_gender else "body_composition",
    }


def _build_rule_context(bodycomp: dict | None, biomechanics: dict | None) -> dict:
    profile = _resolve_profile(bodycomp, biomechanics)
    gender = profile["gender"]

    body_fat = _to_float(_deep_get(bodycomp, "measurements.body_fat_percentage"))
    smi = _to_float(_deep_get(bodycomp, "derived_metrics.SMI"))
    if smi is None:
        smi = _to_float(_deep_get(bodycomp, "measurements.smi_kg_m2"))

    knee = _to_float(_deep_get(biomechanics, "biomechanics.overhead_squat.o3_knee_angle"))
    ankle = _to_float(_deep_get(biomechanics, "biomechanics.overhead_squat.o4_ankle_flex_angle"))
    lower_ai = _to_float(_deep_get(bodycomp, "derived_metrics.AI_lower_pct"))
    upper_ai = _to_float(_deep_get(bodycomp, "derived_metrics.AI_upper_pct"))

    # User-specified default for missing rule variables = normal.
    bodyfat_status = "normal"
    if body_fat is not None:
        bodyfat_status = "high" if body_fat > (25 if gender == "male" else 30) else "normal"

    smi_status = "normal"
    if smi is not None:
        smi_status = "low" if smi < (7.0 if gender == "male" else 5.7) else "normal"

    knee_status = "normal" if knee is None or knee < 95 else "limited"
    ankle_status = "normal" if ankle is None or ankle > 24 else "limited"
    lower_ai_status = "normal" if lower_ai is None or lower_ai < 10 else "warn"
    upper_ai_status = "normal" if upper_ai is None or upper_ai < 10 else "warn"

    parts = [
        profile["age_group"],
        gender,
        bodyfat_status,
        smi_status,
        knee_status,
        ankle_status,
        lower_ai_status,
        upper_ai_status,
    ]

    return {
        **profile,
        "body_fat_percentage": body_fat,
        "SMI": smi,
        "knee_angle": knee,
        "ankle_flex_angle": ankle,
        "AI_lower_pct": lower_ai,
        "AI_upper_pct": upper_ai,
        "bodyfat_status": bodyfat_status,
        "smi_status": smi_status,
        "knee_status": knee_status,
        "ankle_status": ankle_status,
        "lower_ai_status": lower_ai_status,
        "upper_ai_status": upper_ai_status,
        "rule_key": "_".join(parts),
    }


# =============================================================================
# Rule-base loading and exercise expansion
# =============================================================================

def _validate_main_rule_v2(data: dict, rule_key: str, source: Path) -> dict:
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"main_rule 格式錯誤：{source.name} root 必須是 object")

    avoid = data.get("avoid_exercises", [])
    if not isinstance(avoid, list) or any(not isinstance(x, str) for x in avoid):
        raise HTTPException(
            status_code=500,
            detail=f"main_rule 格式錯誤：{source.name} avoid_exercises 必須是 category code 字串陣列",
        )

    plan = data.get("training_plan")
    if not isinstance(plan, dict):
        raise HTTPException(status_code=500, detail=f"main_rule 格式錯誤：{source.name} 缺少 root.training_plan")

    template = plan.get("session_template")
    if not isinstance(template, list):
        raise HTTPException(status_code=500, detail=f"main_rule 格式錯誤：{source.name} training_plan.session_template 必須是 array")

    phases = {}
    for item in template:
        if not isinstance(item, dict):
            continue
        phase = _clean(item.get("phase")).lower()
        if phase:
            phases[phase] = item

    missing = [phase for phase in ("warmup", "main", "cooldown") if phase not in phases]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"main_rule 格式錯誤：{source.name} 缺少 phase: {', '.join(missing)}",
        )

    logger.info("Loaded main_rule v2: %s from %s", rule_key, source)
    return data


def _find_main_rule(rule_key: str) -> dict:
    """
    Main Rule v2:
    - rule_key 以檔名為唯一來源，不要求 JSON 內再重複存 rule_key。
    - 優先找 main_rule/<rule_key>.json。
    - 若未命中，也支援 main_rule 子資料夾遞迴尋找同名檔案。
    """
    direct = MAIN_RULE_DIR / f"{rule_key}.json"
    if direct.exists():
        return _validate_main_rule_v2(_load_json(direct), rule_key, direct)

    if MAIN_RULE_DIR.exists():
        matches = list(MAIN_RULE_DIR.rglob(f"{rule_key}.json"))
        if matches:
            if len(matches) > 1:
                logger.warning("Found multiple main_rule files for %s; using %s", rule_key, matches[0])
            return _validate_main_rule_v2(_load_json(matches[0]), rule_key, matches[0])

    raise HTTPException(
        status_code=404,
        detail=f"找不到 main_rule：{rule_key}.json",
    )


def _resolve_hard_index(questionnaire: dict | None, today_status: dict | None) -> tuple[int, str, str]:
    # Today Status only applies to today; caller already supplies current-day status.
    today_status = _safe_dict(today_status)
    if today_status:
        status = _clean(today_status.get("intensity_status")).lower()
        mapping = {"light": 1, "moderate": 2, "vigorous": 3}
        if status in mapping:
            return mapping[status], status, "today_status"

    questionnaire = _safe_dict(questionnaire)
    # Mutual-resolution priority defined by user: light > moderate > vigorous.
    if questionnaire.get("prefers_light_intensity") is True:
        return 1, "light", "questionnaire"
    if questionnaire.get("prefers_moderate_intensity") is True:
        return 2, "moderate", "questionnaire"
    if questionnaire.get("prefers_vigorous_intensity") is True:
        return 3, "vigorous", "questionnaire"
    return 2, "moderate", "default"


def _expand_category(category_code: str, hard_index: int, pose_catalog: dict) -> list[str]:
    pose_catalog = _safe_dict(pose_catalog)
    category = _safe_dict(pose_catalog.get(category_code))
    exercises = _safe_list(category.get("fitness"))
    out = []
    for entry in exercises:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        pose_name, pose_hard_index = entry[0], entry[1]
        try:
            if int(pose_hard_index) == int(hard_index):
                out.append(_clean(pose_name))
        except Exception:
            continue
    return _unique(out)


def _expand_category_all(category_code: str, pose_catalog: dict) -> list[str]:
    pose_catalog = _safe_dict(pose_catalog)
    category = _safe_dict(pose_catalog.get(category_code))
    out = []
    for entry in _safe_list(category.get("fitness")):
        if isinstance(entry, (list, tuple)) and entry:
            out.append(_clean(entry[0]))
    return _unique(out)


def _phase_map(training_plan: dict) -> dict[str, dict]:
    training_plan = _safe_dict(training_plan)
    result = {}
    for item in _safe_list(training_plan.get("session_template")):
        if isinstance(item, dict) and item.get("phase"):
            result[_clean(item["phase"]).lower()] = item
    return result


def _parse_pairs(raw: Any) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        return []
    pairs = []
    for i in range(0, len(raw), 2):
        pose = _clean(raw[i]) if i < len(raw) else ""
        comment = _clean(raw[i + 1]) if i + 1 < len(raw) else ""
        if pose:
            pairs.append((pose, comment))
    return pairs


def _build_issue_pool(
    biomechanics: dict | None,
    issue_catalog: dict,
    blocked: set[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Randomly select up to two safety-eligible exercises for each active issue."""
    issue_counts = _safe_dict(_deep_get(biomechanics, "issue_counts", {}))
    issue_catalog = _safe_dict(issue_catalog)
    blocked = blocked or set()
    pool: list[str] = []
    comments: list[dict] = []

    for issue_type in ("dynamic", "static"):
        counts = _safe_dict(issue_counts.get(issue_type))
        catalog_group = _safe_dict(issue_catalog.get(issue_type))
        if not counts:
            continue

        for issue_key, count in counts.items():
            numeric_count = _to_float(count) or 0
            if numeric_count <= 0:
                continue

            candidates = []
            seen_pose = set()
            for pose, comment in _parse_pairs(catalog_group.get(issue_key)):
                if not pose or pose in seen_pose or pose in blocked:
                    continue
                seen_pose.add(pose)
                candidates.append((pose, comment))

            if not candidates:
                continue

            select_count = 2 if len(candidates) >= 2 else 1
            selected = random.sample(candidates, k=select_count)
            issue_problem = f"{issue_type} - {issue_key}"

            for pose, comment in selected:
                pool.append(pose)
                comments.append({
                    "pose": pose,
                    "issue_problem": issue_problem,
                    "reason": comment,
                    "issue_type": issue_type,
                    "issue_key": issue_key,
                    "issue_count": numeric_count,
                })

    pool = _unique(pool)
    seen = set()
    comments_unique = []
    for row in comments:
        key = (row["pose"], row["reason"], row["issue_type"], row["issue_key"])
        if key not in seen:
            seen.add(key)
            comments_unique.append(row)

    type_order = {"dynamic": 0, "static": 1}
    comments_unique.sort(key=lambda row: (
        type_order.get(row.get("issue_type"), 99),
        _clean(row.get("issue_key")).lower(),
        _clean(row.get("pose")).lower(),
    ))
    return pool, comments_unique

def _build_history_block_pool(history: dict | None, today_status: dict | None, history_catalog: dict) -> tuple[list[str], list[str]]:
    history = _safe_dict(history)
    history_catalog = _safe_dict(history_catalog)
    true_flags = [key for key, value in history.items() if value is True]

    # Today's selected body areas act as temporary safety blocks.
    # They are stored with the same discomfort_* canonical keys as history.
    today_status = _safe_dict(today_status)
    today_body = _safe_dict(today_status.get("body_status"))
    true_flags.extend([key for key, value in today_body.items() if value is True])
    true_flags = _unique(true_flags)

    pool = []
    for key in true_flags:
        for pose, _comment in _parse_pairs(history_catalog.get(key)):
            pool.append(pose)
    return _unique(pool), true_flags


def _parse_milestones(note: str | None) -> list[dict]:
    """
    支援：
      第4週評估...
      第4週末評估...
      第4週目標...
      第4週末目標...
    UI 統一只顯示 W4 / W8 / W12，不把「末評估」殘留在 target 前面。
    """
    text = _clean(note)
    if not text:
        return []

    pattern = r"第\s*(\d+)\s*週(?:\s*末)?(?:\s*(?:評估|目標))?[：:]?"
    matches = list(re.finditer(pattern, text))
    if matches:
        out = []
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            target = text[start:end].strip("；;，,。 ")
            if target:
                out.append({"week": int(match.group(1)), "target": target})
        return out

    chunks = [c.strip() for c in re.split(r"[；;]", text) if c.strip()]
    return [{"week": None, "target": chunk} for chunk in chunks]


EMPTY_EXERCISE_MESSAGE = "今日身體狀況沒有適合動作，請確認學員身體狀況"


def _build_training_plan(
    rule_data: dict,
    pose_catalog: dict,
    issue_catalog: dict,
    history_catalog: dict,
    biomechanics: dict | None,
    history: dict | None,
    questionnaire: dict | None,
    today_status: dict | None,
) -> dict:
    # Main Rule v2: training_plan 位於 JSON root。
    rule_data = _safe_dict(rule_data)
    pose_catalog = _safe_dict(pose_catalog)
    issue_catalog = _safe_dict(issue_catalog)
    history_catalog = _safe_dict(history_catalog)
    training_plan = _safe_dict(rule_data.get("training_plan"))
    phases = _phase_map(training_plan)

    hard_index, intensity_status, intensity_source = _resolve_hard_index(questionnaire, today_status)

    raw_phase_pools: dict[str, list[str]] = {}
    phase_payload: dict[str, dict] = {}
    for phase_name in ("warmup", "main", "cooldown"):
        phase = _safe_dict(phases.get(phase_name))
        pool = []
        for category_code in _safe_list(phase.get("category_codes")):
            pool.extend(_expand_category(_clean(category_code), hard_index, pose_catalog))
        raw_phase_pools[phase_name] = _unique(pool)
        phase_payload[phase_name] = {
            "duration_min": phase.get("duration_min"),
            "category_codes": _safe_list(phase.get("category_codes")),
            "note": phase.get("note") or "",
        }

    # Main Rule v2: avoid_exercises = ["A3", "B3", ...]
    # Avoid 永遠忽略 hard_index，該 category 下所有動作都加入 block pool。
    main_avoid_pool = []
    avoid_details = []
    for raw_code in _safe_list(rule_data.get("avoid_exercises")):
        category_code = _clean(raw_code)
        if not category_code:
            continue
        expanded = _expand_category_all(category_code, pose_catalog)
        main_avoid_pool.extend(expanded)
        avoid_details.append(
            {
                "category_code": category_code,
                "exercises": expanded,
            }
        )
    main_avoid_pool = _unique(main_avoid_pool)

    history_block_pool, effective_block_flags = _build_history_block_pool(history, today_status, history_catalog)
    blocked = set(history_block_pool) | set(main_avoid_pool)

    # Sample issue exercises only after safety exclusions are known.
    issue_pool, issue_comments = _build_issue_pool(biomechanics, issue_catalog, blocked)

    final_warmup = [x for x in raw_phase_pools["warmup"] if x not in blocked]
    final_cooldown = [x for x in raw_phase_pools["cooldown"] if x not in blocked]
    merged_main = _unique(raw_phase_pools["main"] + issue_pool)
    final_main = [x for x in merged_main if x not in blocked]

    final_main_set = set(final_main)
    issue_rows = [row for row in issue_comments if row["pose"] in final_main_set]


    risk_muscles = []
    for raw in (biomechanics or {}).get("risk_muscles") or []:
        try:
            code = int(float(raw))
        except (TypeError, ValueError):
            continue
        if code > 0:
            risk_muscles.append(code)
    risk_muscles = sorted(set(risk_muscles))

    warmup_exercises = _unique(final_warmup)
    main_exercises = _unique(final_main)
    cooldown_exercises = _unique(final_cooldown)

    return {
        "hard_index": hard_index,
        "risk_muscles": risk_muscles,
        "risk_muscle_date": _record_display_date(biomechanics or {}),
        "intensity_status": intensity_status,
        "intensity_source": intensity_source,
        "weekly_structure": training_plan.get("weekly_structure") or {},
        "warmup": {
            **phase_payload["warmup"],
            "exercises": warmup_exercises,
            "empty_message": EMPTY_EXERCISE_MESSAGE if not warmup_exercises else None,
        },
        "main": {
            **phase_payload["main"],
            "exercises": main_exercises,
            "empty_message": EMPTY_EXERCISE_MESSAGE if not main_exercises else None,
            "issue_exercises": issue_rows,
        },
        "cooldown": {
            **phase_payload["cooldown"],
            "exercises": cooldown_exercises,
            "empty_message": EMPTY_EXERCISE_MESSAGE if not cooldown_exercises else None,
        },
        "milestone_note": training_plan.get("milestone_note") or "",
        "milestones": _parse_milestones(training_plan.get("milestone_note")),
        "progression": _deep_get(training_plan, "weekly_structure.progression", ""),
        "debug": {
            "main_warmup_pool": raw_phase_pools["warmup"],
            "main_main_pool": raw_phase_pools["main"],
            "main_cooldown_pool": raw_phase_pools["cooldown"],
            "issue_count_suggested_pool": issue_pool,
            "history_block_pool": history_block_pool,
            "main_avoid_exercise_pool": main_avoid_pool,
            "effective_block_flags": effective_block_flags,
            "avoid_details": avoid_details,
        },
    }


# =============================================================================
# Questionnaire presentation and communication model
# =============================================================================

COMMUNICATION_STYLES = [
    {
        "id": "achievement_connection",
        "label": "成果連結型",
        "instruction": "從已累積的成果或穩定進展切入，再連結到會員在意的生活價值。",
    },
    {
        "id": "life_connection",
        "label": "生活情境型",
        "instruction": "從旅遊、外出、家人活動、日常自主或生活節奏等已知偏好切入。",
    },
    {
        "id": "confidence_building",
        "label": "信心建立型",
        "instruction": "著重會員已做到的事情與持續累積，讓對話提升信心而非製造壓力。",
    },
    {
        "id": "supportive_companion",
        "label": "陪伴鼓勵型",
        "instruction": "用有陪伴感但不幼兒化的方式，讓會員感受到教練有看見他的投入與節奏。",
    },
    {
        "id": "goal_forward",
        "label": "目標推進型",
        "instruction": "從會員既有目標切入，強調下一小步與持續性，不做過度承諾。",
    },
    {
        "id": "casual_interaction",
        "label": "輕鬆互動型",
        "instruction": "用自然、好接話的方式開啟對話，保持專業但降低制式感。",
    },
]


# Questionnaire-driven interaction profiles.
# Tone is no longer freely invented by the LLM: the selected profile determines
# the 3 tone tags and the interaction guidance. When several preferences are
# selected, recent interaction profiles are avoided where possible so the
# experience can vary without ignoring the questionnaire.
INTERACTION_PROFILES = {
    "communication_prefers_casual_chat": {
        "id": "casual_chat",
        "label": "輕鬆聊天",
        "tone_tags": ["輕鬆", "自然", "有互動感"],
        "guidance": "以自然對話與容易接話的方式互動，少用報告式口吻，保留專業但降低制式感。",
        "style_ids": ["casual_interaction", "life_connection", "supportive_companion"],
    },
    "communication_prefers_quiet_focus": {
        "id": "quiet_focus",
        "label": "安靜專注",
        "tone_tags": ["沉穩", "精簡", "少干擾"],
        "guidance": "少聊天、句子短而明確，避免連續追問或過多情緒性語句，讓會員保有專注空間。",
        "style_ids": ["confidence_building", "achievement_connection", "goal_forward"],
    },
    "communication_prefers_data_progress": {
        "id": "data_progress",
        "label": "數據進度",
        "tone_tags": ["具體", "數據導向", "成果可視化"],
        "guidance": "可使用簡短數據或前後變化支持溝通，但要把數字翻成會員容易理解的生活意義。",
        "style_ids": ["achievement_connection", "goal_forward", "confidence_building"],
    },
    "communication_prefers_encouragement": {
        "id": "encouragement",
        "label": "鼓勵回饋",
        "tone_tags": ["溫暖", "鼓勵", "有支持感"],
        "guidance": "先看見會員已做到的部分，以正向回饋建立信心，不用比較或施加表現壓力。",
        "style_ids": ["supportive_companion", "confidence_building", "life_connection"],
    },
    "communication_prefers_challenge": {
        "id": "challenge",
        "label": "挑戰目標",
        "tone_tags": ["積極", "目標感", "有挑戰性"],
        "guidance": "可把下一小步說得更有目標感與挑戰感，但仍需具體、可達成，不以壓迫或競爭刺激。",
        "style_ids": ["goal_forward", "achievement_connection", "confidence_building"],
    },
    "communication_prefers_step_by_step": {
        "id": "step_by_step",
        "label": "循序說明",
        "tone_tags": ["清楚", "有條理", "循序漸進"],
        "guidance": "一次說一個重點，使用清楚順序與簡短步驟，避免資訊一次堆疊太多。",
        "style_ids": ["goal_forward", "confidence_building", "achievement_connection"],
    },
    "communication_prefers_slow_pace_explanation": {
        "id": "slow_pace",
        "label": "慢節奏說明",
        "tone_tags": ["耐心", "慢節奏", "重點清楚"],
        "guidance": "放慢資訊節奏，每次只講必要重點，給會員理解與回應的時間，不急著補充太多內容。",
        "style_ids": ["supportive_companion", "confidence_building", "life_connection"],
    },
    "communication_prefers_demonstration": {
        "id": "demonstration_first",
        "label": "示範優先",
        "tone_tags": ["直觀", "簡潔", "示範優先"],
        "guidance": "說法保持精簡直觀，先用『我先示範／讓你先看重點』的表達方向，再補少量口語重點；不得在文字中提供具體運動動作。",
        "style_ids": ["confidence_building", "casual_interaction", "goal_forward"],
    },
    "communication_prefers_variety": {
        "id": "variety",
        "label": "喜歡變化",
        "tone_tags": ["新鮮", "彈性", "有變化"],
        "guidance": "避免每天使用相同開場與比喻，可更換安全的生活切入角度與說話節奏。",
        "style_ids": ["life_connection", "casual_interaction", "goal_forward", "achievement_connection"],
    },
    "communication_prefers_routine": {
        "id": "routine",
        "label": "固定節奏",
        "tone_tags": ["穩定", "熟悉", "可預期"],
        "guidance": "維持熟悉、可預期的溝通結構，先回顧已知進度，再說今天重點，避免突然改變表達方式。",
        "style_ids": ["achievement_connection", "confidence_building", "supportive_companion"],
    },
    "prefers_exercise_alone": {
        "id": "independent",
        "label": "自主互動",
        "tone_tags": ["尊重空間", "精簡", "自主感"],
        "guidance": "尊重會員自主空間，以必要、精簡資訊為主，不要求持續聊天或情緒回應。",
        "style_ids": ["confidence_building", "achievement_connection", "goal_forward"],
    },
    "prefers_one_on_one_coaching": {
        "id": "one_on_one",
        "label": "一對一互動",
        "tone_tags": ["專注", "個別化", "有陪伴感"],
        "guidance": "讓會員感受到教練有記得他的偏好與進度，用一對一、具體但不壓迫的方式回應。",
        "style_ids": ["supportive_companion", "confidence_building", "life_connection"],
    },
    "prefers_small_group": {
        "id": "small_group",
        "label": "小團體互動",
        "tone_tags": ["親切", "互動", "有參與感"],
        "guidance": "保持親切與參與感，可使用容易接話的說法，但不要把會員拿來與其他人比較。",
        "style_ids": ["casual_interaction", "supportive_companion", "life_connection"],
    },
    "prefers_large_group_class": {
        "id": "group_energy",
        "label": "團體互動",
        "tone_tags": ["活力", "帶動", "群體感"],
        "guidance": "語氣可以更有精神與帶動感，但仍針對會員本人說話，不用競賽或同儕壓力刺激。",
        "style_ids": ["casual_interaction", "goal_forward", "supportive_companion"],
    },
    "prefers_with_family_or_friends": {
        "id": "social_companion",
        "label": "陪伴互動",
        "tone_tags": ["親切", "生活化", "陪伴感"],
        "guidance": "可從陪伴、一起活動與生活情境切入，但只有問卷明確支持時才提及家人或朋友。",
        "style_ids": ["life_connection", "supportive_companion", "casual_interaction"],
    },
}

DEFAULT_INTERACTION_PROFILE = {
    "id": "balanced",
    "label": "自然互動",
    "tone_tags": ["自然", "清楚", "有支持感"],
    "guidance": "使用自然、清楚且不施壓的方式互動；資料不足時不自行推測會員偏好。",
    "style_ids": [x["id"] for x in COMMUNICATION_STYLES],
}

INTENSITY_LABELS = {
    "light": "疲勞",
    "moderate": "正常",
    "vigorous": "加強鍛鍊",
}


def _selected_labels(questionnaire: dict | None, predicate=None) -> list[str]:
    questionnaire = _safe_dict(questionnaire)
    out = []
    for key, value in questionnaire.items():
        if value is not True:
            continue
        if predicate and not predicate(key):
            continue
        out.append(QUESTIONNAIRE_LABELS.get(key, key))
    return _unique(out)


def _fetch_communication_outputs(actor: Actor, branch_id: str, used_id: str) -> list[dict]:
    rows = _query_rows(
        "CoachOutputs",
        [
            ("company_id", "==", actor.company_id),
            ("branch_id", "==", branch_id),
            ("used_id", "==", used_id),
            ("type", "==", "communication"),
        ],
    )
    rows.sort(key=_record_sort_time, reverse=True)
    return rows


def _communication_status(today_status: dict | None) -> tuple[str, str]:
    """Communication state defaults to normal until coach explicitly saves Today Status."""
    today_status = _safe_dict(today_status)
    if today_status:
        value = _clean(today_status.get("intensity_status")).lower()
        if value in INTENSITY_LABELS:
            return value, "today_status"
    return "moderate", "default"


def _safe_communication_profile(
    profile: dict,
    questionnaire: dict | None,
    bodycomp: dict | None,
    biomechanics: dict | None,
    records: list[dict],
    today_status: dict | None,
) -> dict:
    q = _safe_dict(questionnaire)

    def keys(*prefixes):
        return _selected_labels(q, lambda key: any(key.startswith(p) for p in prefixes))

    body_rows = _by_type(records, "body_composition")[:2]
    bio_rows = _by_type(records, "biomechanics")[:2]

    body_current = _to_float(_deep_get(bodycomp, "scores.coachos.value"))
    body_previous = _to_float(_deep_get(body_rows[1], "scores.coachos.value")) if len(body_rows) > 1 else None
    movement_current = _to_float(_deep_get(biomechanics, "scores.weighted_score"))
    movement_previous = _to_float(_deep_get(bio_rows[1], "scores.weighted_score")) if len(bio_rows) > 1 else None

    def trend(cur, prev):
        if cur is None or prev is None:
            return None
        diff = round(cur - prev, 2)
        if diff > 0:
            return f"上升 {diff}"
        if diff < 0:
            return f"下降 {abs(diff)}"
        return "持平"

    comm_status, comm_status_source = _communication_status(today_status)

    return {
        "age": profile.get("age"),
        "age_group": profile.get("age_group"),
        "gender": profile.get("gender"),
        "today_status": {
            "value": comm_status,
            "label": INTENSITY_LABELS[comm_status],
            "source": comm_status_source,
        },
        "goals": keys("goal_"),
        "exercise_preferences": keys(
            "likes_",
            "prefers_exercise_",
            "prefers_one_on_one",
            "prefers_small_group",
            "prefers_large_group",
            "prefers_with_family",
        ),
        "communication_preferences": keys("communication_"),
        "communication_preference_keys": [
            key for key, value in q.items()
            if value is True and key.startswith("communication_")
        ],
        "interaction_preference_keys": [
            key for key in (
                "prefers_exercise_alone",
                "prefers_one_on_one_coaching",
                "prefers_small_group",
                "prefers_large_group_class",
                "prefers_with_family_or_friends",
            )
            if q.get(key) is True
        ],
        "motivation_preferences": keys("motivation_", "motivated_"),
        "travel_preferences": keys("travel_", "likes_travel"),
        "leisure_preferences": keys("leisure_"),
        "self_description": keys(
            "self_description_",
            "comfortable_",
            "likes_app_",
            "prefers_printed_",
            "prefers_large_text",
        ),
        "body_score": {
            "current": body_current,
            "previous": body_previous,
            "trend": trend(body_current, body_previous),
        },
        "movement_score": {
            "current": movement_current,
            "previous": movement_previous,
            "trend": trend(movement_current, movement_previous),
        },
    }


def _resolve_interaction_profile(safe_profile: dict, recent_outputs: list[dict]) -> dict:
    """Choose an interaction profile from the member's questionnaire preferences.

    If multiple preferences are selected, avoid the most recently used interaction
    profiles where possible. This preserves variety while still staying inside the
    user's declared preferences.
    """
    keys = _unique(
        (safe_profile.get("communication_preference_keys") or [])
        + (safe_profile.get("interaction_preference_keys") or [])
    )
    candidates = [INTERACTION_PROFILES[key] for key in keys if key in INTERACTION_PROFILES]
    if not candidates:
        return dict(DEFAULT_INTERACTION_PROFILE)

    recent_ids = [
        _clean(x.get("interaction_type_id"))
        for x in recent_outputs[:2]
        if _clean(x.get("interaction_type_id"))
    ]
    available = [x for x in candidates if x["id"] not in recent_ids]
    if not available:
        available = candidates
    return dict(secrets.choice(available))


def _choose_communication_style(recent_outputs: list[dict], interaction: dict) -> dict:
    recent_ids = [
        _clean(x.get("communication_style_id"))
        for x in recent_outputs[:2]
        if _clean(x.get("communication_style_id"))
    ]
    preferred_ids = set(interaction.get("style_ids") or [])
    preferred = [x for x in COMMUNICATION_STYLES if not preferred_ids or x["id"] in preferred_ids]
    available = [x for x in preferred if x["id"] not in recent_ids]
    if not available:
        available = preferred or COMMUNICATION_STYLES
    return secrets.choice(available)


def _recent_message_context(recent_outputs: list[dict]) -> list[dict]:
    out = []
    for row in recent_outputs[:2]:
        content = _safe_dict(row.get("communication"))
        out.append(
            {
                "style": row.get("communication_style_label"),
                "communication_advice": _clean(content.get("communication_advice"))[:180],
                "coach_script": _clean(content.get("coach_script"))[:120],
            }
        )
    return out


def _fallback_communication(safe_profile: dict, style: dict, interaction: dict) -> dict:
    status = _deep_get(safe_profile, "today_status.value", "moderate")
    goals = safe_profile.get("goals") or []
    travel = safe_profile.get("travel_preferences") or []

    if status == "light":
        advice = "今天先把溝通重點放在節奏與安心感，肯定會員願意持續出席，不需要用表現或分數增加壓力。可以用他在意的生活目標作為連結，讓他知道穩定累積本身就是有效進展。"
        script = "今天我們照你的狀態走就好，不用急著追進度。能穩定把這次完成，就是很好的累積，我們把節奏顧好，後面會更好接上。"
    elif status == "vigorous":
        advice = "今天可以用較有目標感的方式溝通，把會員目前的穩定表現連結到下一階段成果，同時保持具體與可達成。重點是讓會員感受到挑戰來自累積，而不是被要求一次做到更多。"
        script = "你今天的狀態不錯，我們可以把目標再往前推一點。前面的累積已經有基礎，今天就把這股感覺延續下去。"
    else:
        advice = "今天適合用穩定推進的語氣，先肯定會員目前持續累積的成果，再把評估變化連結到他在意的日常價值。不要只報數字，而是讓會員知道規律投入正在逐步變成更有把握的生活感受。"
        script = "今天的節奏很適合繼續穩穩累積，我們把前面的進步延續下去，讓每一次的改變慢慢變成你日常活動時更有把握的感覺。"

    if travel:
        advice += " 若要延伸話題，可自然帶到旅遊、走行程或外出活動的自主感。"
    elif goals:
        advice += f" 可優先連結會員在意的「{goals[0]}」，讓今天的對話更有個人意義。"

    tags = list(interaction.get("tone_tags") or DEFAULT_INTERACTION_PROFILE["tone_tags"])

    return {
        "communication_advice": advice,
        "coach_script": script,
        "tone_tags": tags,
        "tone_guidance": interaction.get("guidance") or DEFAULT_INTERACTION_PROFILE["guidance"],
        "interaction_type_id": interaction.get("id"),
        "interaction_type_label": interaction.get("label"),
        "source": "fallback",
        "communication_style_id": style["id"],
        "communication_style_label": style["label"],
    }


def _generate_communication(
    safe_profile: dict,
    recent_outputs: list[dict],
    metric_context: dict[str, Any] | None = None,
) -> dict:
    interaction = _resolve_interaction_profile(safe_profile, recent_outputs)
    style = _choose_communication_style(recent_outputs, interaction)
    recent_context = _recent_message_context(recent_outputs)
    prompt = f"""
你是 CoachOS 的「教練溝通輔助模型」。

本次自動選定溝通風格：{style['label']}
風格指令：{style['instruction']}

本次互動類型：{interaction['label']}
固定語氣標籤：{"、".join(interaction['tone_tags'])}
互動指令：{interaction['guidance']}

任務：根據輸入的非醫療會員資料與今日狀態，提供教練可直接使用的溝通建議、說法與語氣。

今日狀態處理：
- 疲勞：降低表現壓力、回應節奏與安心感，不要用挑戰性語言逼迫表現。
- 正常：穩定推進、肯定規律累積、連結日常價值。
- 加強鍛鍊：可增加目標感與成果感，但不要過度承諾或製造焦慮。

多樣化要求：
- 最近兩次內容會一併提供。避免沿用相同開場句、相同主軸與高度相似的 coach_script。
- 優先換一個安全且有資料支持的切入角度，例如成果、日常情境、信心、陪伴、目標或自然互動。
- 不可為了變化而捏造會員沒有提供的興趣、家庭、旅行或生活情境。

嚴格限制：
- 不得出現疾病名稱、症狀描述、診斷、醫療判斷、手術、疼痛推論、治療或復健建議。
- 不得提供運動處方、動作名稱或訓練動作建議。
- 不得猜測輸入不存在的興趣、個性或生活狀況。
- 年齡只能用來調整資訊密度與表達方式，不得用年齡推論能力。
- 不要幼兒化、不要說教、不要製造焦慮、不要過度承諾。

溝通原則：
- communication_advice 與 coach_script 必須明顯符合「本次互動類型」與「互動指令」，不能只把互動類型寫在 tone_tags。
- tone_tags 不得自行改寫，語義必須等同固定語氣標籤。
- 優先把評估進步轉換成會員能感受到的日常價值，例如生活自主、旅遊、走行程、陪家人活動、體力、穩定感、活動信心；僅能使用輸入中有支持的情境。
- 若會員偏好看數據，可適度提到評估進步；否則優先生活化說法，不主動報一堆數字。
- 若資料不足，用一般且安全的正向進度回饋。

輸出要求：
- communication_advice：80～140個繁體中文字，約為舊版 1.5 倍以上。
- coach_script：45～90個繁體中文字，教練可直接說出口。
- tone_tags：恰好3個短標籤。
- tone_guidance：30～60個繁體中文字。
- 繁體中文、台灣用語。
"""

    payload = {
        "member_context": safe_profile,
        "recent_communication_to_avoid_repeating": recent_context,
    }

    try:
        client = app.state.genai_client

        response = _generate_content_with_retry(
            client,
            model=GEMINI_MODEL,
            contents=[
                prompt,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CommunicationOutput,
                temperature=0.72,
            ),
            operation="coach_communication",
            metric_context={
                **(metric_context or {}),
                "prompt_version": "coach_communication_v6_20260929",
                "schema_output_version": "CommunicationOutput_v1",
                "image_count": 0,
                "image_bytes": 0,
            },
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, CommunicationOutput):
            result = parsed.model_dump()
        elif isinstance(parsed, dict):
            result = CommunicationOutput.model_validate(parsed).model_dump()
        else:
            result = CommunicationOutput.model_validate_json(response.text).model_dump()
        # Enforce questionnaire-derived interaction tone after generation so the
        # UI cannot drift back to a generic "具體／成果導向／清楚" tone.
        result["tone_tags"] = list(interaction.get("tone_tags") or DEFAULT_INTERACTION_PROFILE["tone_tags"])
        result["tone_guidance"] = interaction.get("guidance") or DEFAULT_INTERACTION_PROFILE["guidance"]
        result.update(
            {
                "source": "gemini",
                "communication_style_id": style["id"],
                "communication_style_label": style["label"],
                "interaction_type_id": interaction.get("id"),
                "interaction_type_label": interaction.get("label"),
            }
        )
        return result
    except Exception as exc:
        logger.exception("Communication generation failed: %s", exc)
        fallback = _fallback_communication(safe_profile, style, interaction)
        fallback["warning"] = str(exc)
        return fallback


def _save_communication_output(
    actor: Actor,
    branch_id: str,
    used_id: str,
    user_name: str,
    communication: dict,
    safe_profile: dict,
) -> str:
    local_now = datetime.now(TAIPEI_TZ)
    suffix = secrets.token_hex(2)
    doc_id = "_".join(
        [
            _sanitize_doc_id_part(actor.company_id, "COMPANY"),
            _sanitize_doc_id_part(branch_id, "BRANCH"),
            _sanitize_doc_id_part(actor.coach_id, "COACH"),
            _sanitize_doc_id_part(used_id, "USER"),
            "communication",
            local_now.strftime("%Y%m%d_%H%M%S_%f"),
            suffix,
        ]
    )
    record = {
        "company_id": actor.company_id,
        "branch_id": branch_id,
        "coach_id": actor.coach_id,
        "used_id": used_id,
        "user_name": user_name or used_id,
        "type": "communication",
        "record_date": local_now.strftime("%Y-%m-%d"),
        "recorded_at_local": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "today_intensity": _deep_get(safe_profile, "today_status.value", "moderate"),
        "communication_style_id": communication.get("communication_style_id"),
        "communication_style_label": communication.get("communication_style_label"),
        "interaction_type_id": communication.get("interaction_type_id"),
        "interaction_type_label": communication.get("interaction_type_label"),
        "communication": {
            "communication_advice": communication.get("communication_advice"),
            "coach_script": communication.get("coach_script"),
            "tone_tags": communication.get("tone_tags") or [],
            "tone_guidance": communication.get("tone_guidance"),
            "source": communication.get("source"),
        },
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    _db().collection("CoachOutputs").document(doc_id).set(record)
    return doc_id


def _communication_from_output_record(row: dict | None) -> dict | None:
    if not row:
        return None
    content = dict(_safe_dict(row.get("communication")))
    if not content:
        return None
    content["communication_style_id"] = row.get("communication_style_id")
    content["communication_style_label"] = row.get("communication_style_label")
    content["interaction_type_id"] = row.get("interaction_type_id")
    content["interaction_type_label"] = row.get("interaction_type_label")
    content["today_intensity"] = row.get("today_intensity")
    content["generated_date"] = row.get("record_date")
    content["generated_by"] = row.get("coach_id")
    return content


def _ensure_today_communication(
    actor: Actor,
    branch_id: str,
    used_id: str,
    user_name: str,
    safe_profile: dict,
) -> dict:
    recent = _fetch_communication_outputs(actor, branch_id, used_id)
    today = datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d")
    if recent and _record_display_date(recent[0]) == today:
        existing = _communication_from_output_record(recent[0])
        # v5 and older communication records do not contain questionnaire-driven
        # interaction metadata. Treat them as stale once v6 is deployed so the
        # first refresh regenerates a tone that follows the member questionnaire.
        if existing and existing.get("interaction_type_id"):
            return existing

    generated = _generate_communication(
        safe_profile,
        recent,
        {
            "company_id": actor.company_id,
            "branch_id": branch_id,
            "coach_id": actor.coach_id,
            "used_id": used_id,
        },
    )
    _save_communication_output(actor, branch_id, used_id, user_name, generated, safe_profile)
    generated["generated_date"] = today
    generated["generated_by"] = actor.coach_id
    generated["today_intensity"] = _deep_get(safe_profile, "today_status.value", "moderate")
    return generated

# =============================================================================
# Assessment pages
# =============================================================================

def _trend_rows(records: list[dict], record_type: str, value_path: str, limit: int = 5) -> list[dict]:
    out = []
    for row in _by_type(records, record_type):
        value = _to_float(_deep_get(row, value_path))
        if value is None:
            continue
        out.append(
            {
                "date": _record_display_date(row),
                "value": value,
                "coach_id": row.get("coach_id"),
                "doc_id": row.get("_doc_id"),
            }
        )
        if len(out) >= limit:
            break
    return list(reversed(out))


def _is_abnormal_by_rule(value: float | None, rule: dict | None) -> bool | None:
    """Return True only when a measured value violates the reference rule."""
    if value is None or not rule:
        return None
    kind = rule.get("type")
    if kind == "lt":
        return value >= float(rule["value"])
    if kind == "lte":
        return value > float(rule["value"])
    if kind == "gt":
        return value <= float(rule["value"])
    if kind == "gte":
        return value < float(rule["value"])
    if kind == "range":
        return value < float(rule["low"]) or value > float(rule["high"])
    return None


def _reference_lower_bound(reference: str | None) -> float | None:
    """Extract the lower bound used by the Gait lower-bound-only rule."""
    ref = _clean(reference)
    if not ref or ref == "-":
        return None
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", ref)
    if not nums:
        return None
    # For both a range (44 - 119) and a > threshold, the first numeric value is
    # the lower bound. Gait intentionally ignores upper bounds by product spec.
    try:
        return float(nums[0])
    except (TypeError, ValueError):
        return None


def _gait_is_below_lower_bound(value: float | None, reference: str | None) -> bool | None:
    if value is None:
        return None
    lower = _reference_lower_bound(reference)
    if lower is None:
        return None
    return value < lower


def _bodycomp_comparisons(body: dict | None, biomechanics: dict | None) -> dict:
    """Build deterministic reference checks for Body Composition values.

    Only values with an explicit product/report specification are graded. Other
    measurements remain visible but are not silently assigned a medical range.
    """
    body = _safe_dict(body)
    measurements = _safe_dict(body.get("measurements"))
    derived = _safe_dict(body.get("derived_metrics"))

    gender = _normalize_gender(body.get("gender"))
    if not gender:
        gender = _normalize_gender(_deep_get(biomechanics or {}, "profile.gender"))

    result: dict[str, dict] = {}

    # PBF 固定使用 CoachOS gender reference。
    # 不讀取報表或歷史資料中的 PBF reference low/high。
    pbf = _to_float(
        measurements.get(
            "body_fat_percentage"
        )
    )

    pbf_low = None
    pbf_high = None

    if gender == "male":
        pbf_low, pbf_high = 10.0, 20.0

    elif gender == "female":
        pbf_low, pbf_high = 18.0, 28.0
    if pbf_low is not None and pbf_high is not None:
        result["measurements.body_fat_percentage"] = {
            "reference": f"{pbf_low:g}% - {pbf_high:g}%",
            "abnormal": None if pbf is None else (pbf < pbf_low or pbf > pbf_high),
        }

    # SMI threshold is the same one used by the rule-key logic.
    if gender:
        smi_min = 7.0 if gender == "male" else 5.7
        for path, value in [
            ("measurements.smi_kg_m2", _to_float(measurements.get("smi_kg_m2"))),
            ("derived_metrics.SMI", _to_float(derived.get("SMI"))),
        ]:
            result[path] = {
                "reference": f">= {smi_min:g} kg/m²",
                "abnormal": None if value is None else value < smi_min,
            }

    for key in ("AI_upper_pct", "AI_lower_pct"):
        value = _to_float(derived.get(key))
        result[f"derived_metrics.{key}"] = {
            "reference": "< 10%",
            "abnormal": None if value is None else value >= 10.0,
        }

    return result


def _comparison_page(records: list[dict], spec_key: str) -> dict:
    spec = ASSESSMENT_SPECS[spec_key]
    recent_desc = _recent_with_section(records, spec["section_path"], 2)
    display_rows = list(reversed(recent_desc))  # older -> newer, left -> right

    dates = [_record_display_date(row) for row in display_rows]
    source_records = [
        {
            "date": _record_display_date(row),
            "coach_id": row.get("coach_id"),
            "doc_id": row.get("_doc_id"),
        }
        for row in display_rows
    ]

    metrics = []
    for metric in spec["metrics"]:
        values = []
        abnormal_values = []
        for row in display_rows:
            value = _to_float(_deep_get(row, f"{spec['section_path']}.{metric['key']}"))
            values.append(value)
            abnormal_values.append(_is_abnormal_by_rule(value, metric.get("rule")))
        metrics.append({**metric, "values": values, "abnormal_values": abnormal_values})

    return {
        "title": spec["title"],
        "image": spec.get("image"),
        "dates": dates,
        "source_records": source_records,
        "metrics": metrics,
    }


def _resolve_height_m(bodycomp: dict | None, biomechanics: dict | None) -> float | None:
    height_m = _to_float(_deep_get(bodycomp, "measurements.height_m"))
    if height_m and height_m > 0:
        return height_m
    height_m = _to_float(_deep_get(biomechanics, "profile.height_m"))
    if height_m and height_m > 0:
        return height_m
    height_cm = _to_float(_deep_get(biomechanics, "profile.height_cm"))
    if height_cm and height_cm > 0:
        return height_cm / 100.0
    return None


def _build_gait_page(records: list[dict], bodycomp: dict | None, biomechanics: dict | None) -> dict:
    recent_desc = _recent_with_section(records, "gait_metrics", 2)
    display_rows = list(reversed(recent_desc))
    dates = [_record_display_date(row) for row in display_rows]

    age = None
    gender = None
    try:
        profile = _resolve_profile(bodycomp, biomechanics)
        age = profile["age"]
        gender = "m" if profile["gender"] == "male" else "f"
    except HTTPException:
        pass

    height_m = _resolve_height_m(bodycomp, biomechanics)
    references = {
        "gait_speed_cm_s": "> 70 cm/s" if age is not None and age < 65 else "-",
        "step_length_cm": "-",
        "cadence_spm": "-",
        "symmetry_pct": "> 90%",
    }
    reference_source = "fixed_under_65" if age is not None and age < 65 else "unavailable"

    if age is not None and age >= 65 and gender and height_m:
        try:
            references.update(build_gait_references(age, gender, height_m))
            reference_source = "gait_eval_prediction_interval"
        except Exception as exc:
            logger.warning("Gait normative calculation failed: %s", exc)
            reference_source = "gait_norm_unavailable"

    metric_defs = [
        {"key": "gait_speed_cm_s", "label": "G-1. 步速 (cm/s)", "unit": "cm/s", "decimals": 1},
        {"key": "step_length_cm", "label": "G-2. 步長 (cm)", "unit": "cm", "decimals": 1},
        {"key": "cadence_spm", "label": "G-3. 步頻 (steps/min)", "unit": "steps/min", "decimals": 1},
        {"key": "symmetry_pct", "label": "G-4. 左右對稱性 (%)", "unit": "%", "decimals": 1},
    ]

    metrics = []
    for metric in metric_defs:
        values = []
        for row in display_rows:
            gait = _safe_dict(row.get("gait_metrics"))
            if metric["key"] == "symmetry_pct":
                ratio = _to_float(gait.get("symmetry_ratio"))
                values.append(round((1.0 - ratio) * 100.0, 1) if ratio is not None else None)
            else:
                values.append(_to_float(gait.get(metric["key"])))
        reference_label = "參考"
        if reference_source == "gait_eval_prediction_interval" and metric["key"] != "symmetry_pct":
            reference_label = "65+ 個人常模"
        reference = references.get(metric["key"], "-")
        metrics.append({
            **metric,
            "reference": reference,
            "reference_label": reference_label,
            "values": values,
            # Gait intentionally grades ONLY the lower bound. Values above a PI
            # upper bound are not marked red.
            "abnormal_values": [_gait_is_below_lower_bound(value, reference) for value in values],
            "comparison_mode": "lower_bound_only",
        })

    return {
        "title": "Gait 步態分析",
        "image": None,
        "dates": dates,
        "source_records": [
            {
                "date": _record_display_date(row),
                "coach_id": row.get("coach_id"),
                "doc_id": row.get("_doc_id"),
            }
            for row in display_rows
        ],
        "metrics": metrics,
        "reference_meta": {
            "source": reference_source,
            "label": (
                "65+ 個人常模"
                if reference_source == "gait_eval_prediction_interval"
                else "固定參考值"
                if reference_source == "fixed_under_65"
                else "參考值"
            ),
            "description": (
                "依年齡、性別與身高校正的 Prediction Interval"
                if reference_source == "gait_eval_prediction_interval"
                else "65 歲以下使用固定 CoachOS Gait 規格"
                if reference_source == "fixed_under_65"
                else "缺少計算常模所需資料"
            ),
            "age": age,
            "gender": gender,
            "height_m": height_m,
        },
    }


def _build_assessment(records: list[dict]) -> dict:
    body = _latest(records, "body_composition")
    bio = _latest(records, "biomechanics")

    body_score = _to_float(_deep_get(body, "scores.coachos.value"))
    movement_score = _to_float(_deep_get(bio, "scores.weighted_score"))

    bodycomp_payload = None
    if body:
        bodycomp_payload = {
            "date": _record_display_date(body),
            "coach_id": body.get("coach_id"),
            "device_type": body.get("device_type"),
            "age": body.get("age"),
            "gender": body.get("gender"),
            "measurements": _safe_dict(body.get("measurements")),
            "segmental_muscle": _safe_dict(body.get("segmental_muscle")),
            "derived_metrics": _safe_dict(body.get("derived_metrics")),
            "scores": _safe_dict(body.get("scores")),
            "comparisons": _bodycomp_comparisons(body, bio),
        }

    return {
        "page1_score": {
            "body_score": body_score,
            "body_score_label": "CoachOS BCS",
            "movement_score": movement_score,
            "movement_score_label": "體測 Weighted Score",
            "body_date": _record_display_date(body),
            "movement_date": _record_display_date(bio),
        },
        "page2_trend": {
            "body": _trend_rows(records, "body_composition", "scores.coachos.value", 5),
            "movement": _trend_rows(records, "biomechanics", "scores.weighted_score", 5),
        },
        "page3_bodycomp": bodycomp_payload,
        "page4_front": _comparison_page(records, "front"),
        "page5_side": _comparison_page(records, "side"),
        "page6_bridge": _comparison_page(records, "bridge"),
        "page7_ohs": _comparison_page(records, "ohs"),
        "page8_bird_dog": _comparison_page(records, "bird_dog"),
        "page9_gait": _build_gait_page(records, body, bio),
    }

def _build_data_status(records: list[dict]) -> dict:
    """Three-state source completeness for MEMBER PROFILE.

    state:
      ready   = latest record contains the expected core payload for that source.
      partial = a record exists, but one or more expected fields/sections are absent or malformed.
      missing = no record of that type exists.

    This is display metadata only. It does NOT change Rule Key behavior:
    missing rule variables still follow the product default and become "normal".
    """

    def item(record_type: str, label: str, checks: list[tuple[str, bool]]) -> dict:
        record = _latest(records, record_type)
        if not record:
            return {
                "key": record_type,
                "label": label,
                "state": "missing",
                "available": False,     # backward compatibility
                "complete": False,
                "date": None,
                "missing_labels": [],
            }

        missing_labels = [label_text for label_text, ok in checks if not ok]
        state = "ready" if not missing_labels else "partial"
        return {
            "key": record_type,
            "label": label,
            "state": state,
            "available": True,          # record exists; keeps older frontends compatible
            "complete": state == "ready",
            "date": _record_display_date(record),
            "missing_labels": missing_labels,
        }

    bio = _latest(records, "biomechanics")
    body = _latest(records, "body_composition")
    questionnaire_record = _latest(records, "questionnaire")
    history_record = _latest(records, "history")

    # 體測：六個主要頁面 + weighted score + static/dynamic issue counts。
    # issue_counts 的 {} 是合法的「本次沒有 issue」，因此只要求它是 dict。
    bio_checks = []
    if bio:
        bio_checks = [
            ("正面", _has_dict_section(bio, "biomechanics.front")),
            ("側面", _has_dict_section(bio, "biomechanics.side")),
            ("肩橋", _has_dict_section(bio, "biomechanics.shoulder_bridge")),
            ("OHS", _has_dict_section(bio, "biomechanics.overhead_squat")),
            ("Bird Dog", _has_dict_section(bio, "biomechanics.swimming")),
            ("步態", _has_dict_section(bio, "gait_metrics")),
            ("Weighted Score", _has_scalar_value(bio, "scores.weighted_score")),
            ("Static Issue", _has_dict_section(bio, "issue_counts.static", allow_empty=True)),
            ("Dynamic Issue", _has_dict_section(bio, "issue_counts.dynamic", allow_empty=True)),
        ]

    # 體組成：只把 Rule/UI 核心欄位列為「完整」判斷，不要求所有設備選配欄位。
    body_checks = []
    if body:
        age_ok = _has_scalar_value(body, "age") or _has_scalar_value(body, "profile.age")
        gender_ok = _has_scalar_value(body, "gender") or _has_scalar_value(body, "profile.gender")
        smi_ok = (
            _has_scalar_value(body, "derived_metrics.SMI")
            or _has_scalar_value(body, "measurements.smi_kg_m2")
        )
        body_checks = [
            ("年齡", age_ok),
            ("性別", gender_ok),
            ("體脂率", _has_scalar_value(body, "measurements.body_fat_percentage")),
            ("SMI", smi_ok),
            ("上肢 AI", _has_scalar_value(body, "derived_metrics.AI_upper_pct")),
            ("下肢 AI", _has_scalar_value(body, "derived_metrics.AI_lower_pct")),
        ]

    # 問卷：有 document 但 questionnaire 不是非空 dict，顯示黃燈。
    questionnaire_checks = []
    if questionnaire_record:
        q = questionnaire_record.get("questionnaire", _MISSING)
        questionnaire_checks = [("問卷內容", isinstance(q, dict) and bool(q))]

    # 歷史資訊：空 dict 可以代表「已收集、沒有勾選任何歷史狀況」，因此仍算完整。
    history_checks = []
    if history_record:
        h = history_record.get("history", _MISSING)
        history_checks = [("歷史資訊內容", isinstance(h, dict))]

    items = [
        item("biomechanics", "體測", bio_checks),
        item("body_composition", "體組成", body_checks),
        item("questionnaire", "問卷", questionnaire_checks),
        item("history", "歷史資訊", history_checks),
    ]

    states = [x["state"] for x in items]
    if all(state == "ready" for state in states):
        overall_state = "ready"
    elif all(state == "missing" for state in states):
        overall_state = "missing"
    else:
        overall_state = "partial"

    return {
        "overall_state": overall_state,
        "all_ready": overall_state == "ready",   # backward compatibility
        "has_partial": any(state == "partial" for state in states),
        "items": items,
    }


def _resolve_display_profile(
    biomechanics: dict | None,
    bodycomp: dict | None,
    questionnaire_record: dict | None,
    history_record: dict | None,
) -> dict:
    """Best-effort profile for UI/communication when collection is incomplete.

    Rule Key source priority is unchanged: biomechanics -> body composition.
    Questionnaire/history are never used to construct a Rule Key.
    """
    candidates = [
        ("biomechanics", _deep_get(biomechanics or {}, "profile.age"), _deep_get(biomechanics or {}, "profile.gender")),
        ("body_composition", (bodycomp or {}).get("age") or _deep_get(bodycomp or {}, "profile.age"), (bodycomp or {}).get("gender") or _deep_get(bodycomp or {}, "profile.gender")),
        ("questionnaire", (questionnaire_record or {}).get("age") or _deep_get(questionnaire_record or {}, "profile.age"), (questionnaire_record or {}).get("gender") or _deep_get(questionnaire_record or {}, "profile.gender")),
        ("history", (history_record or {}).get("age") or _deep_get(history_record or {}, "profile.age"), (history_record or {}).get("gender") or _deep_get(history_record or {}, "profile.gender")),
    ]

    age = None
    age_source = None
    gender = None
    gender_source = None
    for source, raw_age, raw_gender in candidates:
        if age is None:
            parsed_age = _to_int(raw_age)
            if parsed_age is not None:
                age = parsed_age
                age_source = source
        if gender is None:
            parsed_gender = _normalize_gender(raw_gender)
            if parsed_gender:
                gender = parsed_gender
                gender_source = source

    age_group = None
    if age is not None:
        try:
            age_group = _age_group(age)
        except HTTPException:
            age_group = None

    return {
        "age": age,
        "age_group": age_group,
        "gender": gender,
        "age_source": age_source,
        "gender_source": gender_source,
    }


def _try_build_rule_context(bodycomp: dict | None, biomechanics: dict | None) -> tuple[dict | None, str | None]:
    try:
        return _build_rule_context(bodycomp, biomechanics), None
    except HTTPException as exc:
        if exc.status_code == 422:
            return None, str(exc.detail)
        raise


def _unavailable_training_plan(
    reason: str | None,
    biomechanics: dict | None,
    questionnaire: dict | None,
    today_status: dict | None,
) -> dict:
    hard_index, intensity_status, intensity_source = _resolve_hard_index(questionnaire, today_status)
    risk_muscles = []
    for raw in (biomechanics or {}).get("risk_muscles") or []:
        try:
            code = int(float(raw))
        except (TypeError, ValueError):
            continue
        if code > 0:
            risk_muscles.append(code)

    empty_phase = {
        "duration_min": None,
        "category_codes": [],
        "note": "",
        "exercises": [],
        "empty_message": EMPTY_EXERCISE_MESSAGE,
    }
    return {
        "available": False,
        "unavailable_reason": reason or "缺少產生訓練計畫所需資料",
        "hard_index": hard_index,
        "risk_muscles": sorted(set(risk_muscles)),
        "risk_muscle_date": _record_display_date(biomechanics or {}),
        "intensity_status": intensity_status,
        "intensity_source": intensity_source,
        "weekly_structure": {},
        "warmup": dict(empty_phase),
        "main": {**empty_phase, "issue_exercises": []},
        "cooldown": dict(empty_phase),
        "milestone_note": "",
        "milestones": [],
        "progression": "",
        "debug": {"partial_data": True},
    }


# =============================================================================
# Full member payload
# =============================================================================

def _build_member_payload(
    actor: Actor,
    branch_id: str,
    used_id: str,
    include_communication: bool = True,
    communication_override: dict | None = None,
) -> dict:
    branch_id = _enforce_branch(actor, branch_id)
    used_id = _clean(used_id)
    records = _fetch_member_health_records(actor, branch_id, used_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"找不到 {used_id} 的 HealthRecords")

    history_record = _latest(records, "history") or {}
    questionnaire_record = _latest(records, "questionnaire") or {}
    bodycomp = _latest(records, "body_composition") or {}
    biomechanics = _latest(records, "biomechanics") or {}
    today_status = _current_today_status(records)

    history = _safe_dict(history_record.get("history"))
    questionnaire = _safe_dict(questionnaire_record.get("questionnaire"))

    display_profile = _resolve_display_profile(
        biomechanics, bodycomp, questionnaire_record, history_record
    )
    rule_context, rule_unavailable_reason = _try_build_rule_context(bodycomp, biomechanics)

    if rule_context is not None:
        rule_data = _find_main_rule(rule_context["rule_key"])
        pose_catalog = _load_json(POSE_CATALOG_PATH)
        issue_catalog = _load_json(ISSUE_COUNT_PATH)
        history_catalog = _load_json(HISTORY_BLOCK_PATH)
        plan = _build_training_plan(
            rule_data=rule_data,
            pose_catalog=pose_catalog,
            issue_catalog=issue_catalog,
            history_catalog=history_catalog,
            biomechanics=biomechanics,
            history=history,
            questionnaire=questionnaire,
            today_status=today_status,
        )
        plan["available"] = True
        plan["unavailable_reason"] = None
    else:
        plan = _unavailable_training_plan(
            rule_unavailable_reason, biomechanics, questionnaire, today_status
        )

    latest_name = next((_clean(r.get("user_name")) for r in records if _clean(r.get("user_name"))), used_id)
    goals = _selected_labels(questionnaire, lambda key: key.startswith("goal_"))

    effective_body_status = _safe_dict(_safe_dict(today_status).get("body_status"))
    if not effective_body_status:
        effective_body_status = {key: False for key in BODY_STATUS_KEYS}
    _, training_intensity_status, training_intensity_source = _resolve_hard_index(questionnaire, today_status)
    ui_intensity_status, ui_status_source = _communication_status(today_status)

    member = {
        "company_id": actor.company_id,
        "branch_id": branch_id,
        "used_id": used_id,
        "user_name": latest_name,
        "age": (rule_context or display_profile).get("age"),
        "gender": (rule_context or display_profile).get("gender"),
        "goals": goals,
        "data_status": _build_data_status(records),
    }

    safe_profile = _safe_communication_profile(
        rule_context or display_profile,
        questionnaire,
        bodycomp,
        biomechanics,
        records,
        today_status,
    )

    communication = communication_override
    if include_communication and communication is None:
        communication = _ensure_today_communication(
            actor,
            branch_id,
            used_id,
            latest_name,
            safe_profile,
        )

    return _json_safe(
        {
            "viewer": {
                "coach_id": actor.coach_id,
                "display_name": actor.display_name,
                "role": actor.role,
                "company_id": actor.company_id,
                "branch_id": actor.branch_id,
            },
            "member": member,
            "today_status": {
                "exists_today": bool(today_status),
                # UI defaults to normal until an explicit Today Status exists.
                "intensity_status": ui_intensity_status,
                "intensity_source": ui_status_source,
                # Rule Engine still follows the agreed priority:
                # today_status > questionnaire > default.
                "training_intensity_status": training_intensity_status,
                "training_intensity_source": training_intensity_source,
                "body_status": {key: bool(effective_body_status.get(key)) for key in BODY_STATUS_KEYS},
                "body_status_labels": BODY_STATUS_LABELS,
            },
            "rule": rule_context or {
                **display_profile,
                "available": False,
                "rule_key": None,
                "unavailable_reason": rule_unavailable_reason,
            },
            "training": plan,
            "communication": communication,
            "assessment": _build_assessment(records),
            "source_records": {
                "history": history_record.get("_doc_id"),
                "questionnaire": questionnaire_record.get("_doc_id"),
                "body_composition": bodycomp.get("_doc_id"),
                "biomechanics": biomechanics.get("_doc_id"),
                "today_status": (today_status or {}).get("_doc_id"),
            },
        }
    )


# =============================================================================
# API endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "CoachOS Coach API",
        "version": "10.0.0",
        "provider": "vertex_ai",
        "project": getattr(
            app.state,
            "google_cloud_project",
            (
                GOOGLE_CLOUD_PROJECT
                or FIREBASE_PROJECT_ID
            ),
        ),
        "location": GOOGLE_CLOUD_LOCATION,
        "model": GEMINI_MODEL,
        "traffic_mode": "priority_paygo",
    }


@app.get("/api/coach/me")
def get_me(actor: Actor = Depends(require_actor)):
    return {"status": "success", "user": actor.model_dump()}


@app.get("/api/coach/members")
def get_coach_members(actor: Actor = Depends(require_actor)):
    return {"status": "success", "viewer": actor.model_dump(), "members": _visible_members(actor)}


@app.get("/api/coach/member/{used_id}")
def get_coach_member(
    used_id: str,
    branch_id: str = Query(...),
    include_communication: bool = Query(True),
    actor: Actor = Depends(require_actor),
):
    return {
        "status": "success",
        "data": _build_member_payload(actor, _clean(branch_id), _clean(used_id), include_communication),
    }


@app.get("/api/coach/assessment/{used_id}")
def get_coach_assessment(
    used_id: str,
    branch_id: str = Query(...),
    actor: Actor = Depends(require_actor),
):
    records = _fetch_member_health_records(actor, _clean(branch_id), _clean(used_id))
    if not records:
        raise HTTPException(status_code=404, detail="找不到會員評估紀錄")
    return {"status": "success", "assessment": _json_safe(_build_assessment(records))}


@app.post("/api/coach/generate")
def generate_coach_recommendation(payload: TodayStatusRequest, actor: Actor = Depends(require_actor)):
    branch_id = _enforce_branch(actor, payload.branch_id)
    used_id = _clean(payload.used_id)
    _assert_member_access(actor, branch_id, used_id)

    body_status = {key: bool(payload.body_status.get(key)) for key in BODY_STATUS_KEYS}
    local_now = datetime.now(TAIPEI_TZ)
    date_id = local_now.strftime("%Y%m%d")

    # Keep coach_id in the document ID and payload as source metadata. Access
    # and comparisons never use coach_id to split the member's timeline.
    doc_id = "_".join(
        [
            _sanitize_doc_id_part(actor.company_id, "COMPANY"),
            _sanitize_doc_id_part(branch_id, "BRANCH"),
            _sanitize_doc_id_part(actor.coach_id, "COACH"),
            _sanitize_doc_id_part(used_id, "USER"),
            "today_status",
            date_id,
        ]
    )

    record = {
        "company_id": actor.company_id,
        "branch_id": branch_id,
        "coach_id": actor.coach_id,
        "used_id": used_id,
        "user_name": payload.user_name or used_id,
        "type": "today_status",
        "record_date": local_now.strftime("%Y-%m-%d"),
        "recorded_at_local": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "intensity_status": payload.intensity_status,
        "body_status": body_status,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    ref = _db().collection("HealthRecords").document(doc_id)
    snap = ref.get()
    if not snap.exists:
        record["createdAt"] = firestore.SERVER_TIMESTAMP
    ref.set(record, merge=True)

    # Generate a fresh communication note after Today Status is written.
    # Recent styles/messages are excluded where possible to increase variety.
    records = _fetch_member_health_records(actor, branch_id, used_id)
    bodycomp = _latest(records, "body_composition") or {}
    biomechanics = _latest(records, "biomechanics") or {}
    questionnaire_record = _latest(records, "questionnaire") or {}
    questionnaire = _safe_dict(questionnaire_record.get("questionnaire"))
    current_today = _current_today_status(records)
    history_record = _latest(records, "history") or {}
    display_profile = _resolve_display_profile(
        biomechanics, bodycomp, questionnaire_record, history_record
    )
    rule_context, _rule_unavailable_reason = _try_build_rule_context(bodycomp, biomechanics)
    safe_profile = _safe_communication_profile(
        rule_context or display_profile,
        questionnaire,
        bodycomp,
        biomechanics,
        records,
        current_today,
    )
    recent_outputs = _fetch_communication_outputs(actor, branch_id, used_id)
    communication = _generate_communication(
        safe_profile,
        recent_outputs,
        {
            "company_id": actor.company_id,
            "branch_id": branch_id,
            "coach_id": actor.coach_id,
            "used_id": used_id,
        },
    )
    communication_output_doc_id = _save_communication_output(
        actor,
        branch_id,
        used_id,
        payload.user_name or used_id,
        communication,
        safe_profile,
    )
    communication["generated_date"] = local_now.strftime("%Y-%m-%d")
    communication["generated_by"] = actor.coach_id
    communication["today_intensity"] = payload.intensity_status

    data = _build_member_payload(
        actor,
        branch_id,
        used_id,
        include_communication=True,
        communication_override=communication,
    )
    return {
        "status": "success",
        "today_status_doc_id": doc_id,
        "communication_output_doc_id": communication_output_doc_id,
        "data": data,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("coach_api:app", host="0.0.0.0", port=port, reload=False)
