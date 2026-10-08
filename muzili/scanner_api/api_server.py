import os
import json
import logging
import io
import math
import random
import time
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from typing import Literal, Any

from ai_usage_metrics import build_metric_record, new_request_id, save_metric_async


load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "").strip()
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", FIREBASE_PROJECT_ID).strip()
GOOGLE_CLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip() or "global"
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "firebase_key.json").strip()
TAIPEI_TZ = timezone(timedelta(hours=8))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vitruvian 智慧大腦 API",
    description="處理 BIA 身體組成與 CoachOS 標準固定選項問卷影像辨識；結果寫入 Firebase HealthRecords",
)

SCANNER_WEB_ORIGINS = [
    x.strip() for x in os.environ.get(
        "SCANNER_WEB_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",") if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=SCANNER_WEB_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)


# -----------------------------------------------------------------------------
# 1. Firebase + Vertex AI
# -----------------------------------------------------------------------------
def _resolve_service_account_path() -> str | None:
    """Resolve one backend credential file for BOTH Firebase Admin and Vertex AI.

    Public-beta users never see or use this credential. On Render, mount the JSON
    as a Secret File and set FIREBASE_KEY_PATH to its path. The code also exposes
    the same file to Google ADC via GOOGLE_APPLICATION_CREDENTIALS, so no
    `gcloud auth` is needed on the running service.
    """
    raw = Path(FIREBASE_KEY_PATH)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path(__file__).resolve().parent / raw)
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
            "Vertex AI 缺少 GOOGLE_CLOUD_PROJECT / FIREBASE_PROJECT_ID，"
            "且無法從 service account JSON 取得 project_id"
        )

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=GOOGLE_CLOUD_LOCATION,
        http_options=types.HttpOptions(
            api_version="v1",
            headers={
                # CoachOS 即時 Gemini 全部使用 Priority PayGo。
                "X-Vertex-AI-LLM-Request-Type": "shared",
                "X-Vertex-AI-LLM-Shared-Request-Type": "priority",
            },
        ),
    )


# Vertex Standard PayGo/shared capacity may transiently return 429/5xx.
# Retry only the remote HTTP call; schema/Pydantic errors are NOT retried here.
VERTEX_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
VERTEX_RETRY_BASE_DELAYS = (2.0, 5.0, 10.0, 20.0)


def _vertex_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from google-genai exceptions."""
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
    value = getattr(response, "status_code", None) if response is not None else None
    try:
        if value is not None:
            code = int(value)
            if 100 <= code <= 599:
                return code
    except (TypeError, ValueError):
        pass

    # google-genai error text commonly starts with "429 RESOURCE_EXHAUSTED" etc.
    message = str(exc or "")
    for code in VERTEX_RETRYABLE_STATUS_CODES:
        if message.startswith(str(code)) or f"'code': {code}" in message or f'"code": {code}' in message:
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

    Telemetry is intentionally sidecar-only: Firestore metric write failures never
    change the AI response or the existing CoachOS request flow.
    """
    total_attempts = 1 + len(VERTEX_RETRY_BASE_DELAYS)
    last_error: Exception | None = None
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
                    service="scanner_api",
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
                        service="scanner_api",
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

                if retryable:
                    logger.error(
                        "Vertex AI %s exhausted retries: status=%s attempt=%s/%s error=%s",
                        operation, status, attempt, total_attempts, exc,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="AI 服務目前繁忙，請稍後再試；本次資料尚未儲存",
                    ) from exc
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
    project_id = GOOGLE_CLOUD_PROJECT or FIREBASE_PROJECT_ID or _project_id_from_service_account(key_path)

    if key_path:
        # Vertex AI's ADC loader will read this service-account file automatically.
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        logger.info("✅ 使用後端 service account：%s", key_path)

    if not firebase_admin._apps:
        if key_path:
            firebase_admin.initialize_app(
                credentials.Certificate(key_path),
                options={"projectId": project_id} if project_id else None,
            )
        else:
            # Works on Google-managed runtimes with attached service identity / ADC.
            if not project_id:
                raise RuntimeError("Firebase 缺少專案 ID，且未提供 service account credential")
            firebase_admin.initialize_app(options={"projectId": project_id})
            logger.info("Firebase 使用環境提供的 ADC / workload identity")

    app.state.db = firestore.client()
    app.state.google_cloud_project = project_id
    app.state.genai_client = _build_vertex_client(project_id)
    logger.info(
        "🚀 Firebase + Vertex AI 已就緒 project=%s location=%s model=%s",
        project_id, GOOGLE_CLOUD_LOCATION, GEMINI_MODEL
    )


# -----------------------------------------------------------------------------
# 1A. Service health check
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "provider": "vertex_ai",
        "project": getattr(app.state, "google_cloud_project", GOOGLE_CLOUD_PROJECT or FIREBASE_PROJECT_ID),
        "location": GOOGLE_CLOUD_LOCATION,
        "model": GEMINI_MODEL,
    }


@app.api_route("/", methods=["GET", "HEAD"])
def root_health():
    # Render deployment/root probes may call HEAD /. Keep it lightweight.
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# 1B. Scanner authentication / role capability
# -----------------------------------------------------------------------------
class ScannerActor(BaseModel):
    firebase_uid: str
    company_id: str
    branch_id: str | None = None
    coach_id: str
    role: Literal["coach", "manager", "owner"]
    roles: list[str] = Field(default_factory=list)


def _clean_id(value: Any) -> str:
    return str(value or "").strip()


def _default_roles_for_role(role: str) -> list[str]:
    """Backward-compatible capability mapping for old Coaches docs without `roles`."""
    role = _clean_id(role).lower()
    if role == "coach":
        return ["coach"]
    if role == "manager":
        return ["manager", "coach"]
    if role == "owner":
        return ["owner", "coach"]
    return []


def _normalize_roles(data: dict) -> list[str]:
    role = _clean_id(data.get("role")).lower()
    raw_roles = data.get("roles")
    roles: list[str] = []

    if isinstance(raw_roles, list):
        for item in raw_roles:
            value = _clean_id(item).lower()
            if value in {"coach", "manager", "owner"} and value not in roles:
                roles.append(value)

    if not roles:
        roles = _default_roles_for_role(role)

    if role in {"coach", "manager", "owner"} and role not in roles:
        roles.insert(0, role)

    return roles


def _resolve_scanner_actor(decoded: dict) -> ScannerActor:
    """
    ID token proves identity; Firestore Coaches is authoritative for app scope.
    Request form/query fields never grant company / branch / coach permissions.
    """
    uid = _clean_id(decoded.get("uid") or decoded.get("user_id") or decoded.get("sub"))
    if not uid:
        raise HTTPException(status_code=401, detail="Firebase token 缺少 UID")

    snap = app.state.db.collection("Coaches").document(uid).get()
    if snap.exists:
        data = snap.to_dict() or {}
    else:
        # Legacy fallback for old Coaches documents whose doc id was not UID.
        email = _clean_id(decoded.get("email")).lower()
        candidates = []
        if email:
            candidates = list(
                app.state.db.collection("Coaches")
                .where("email", "==", email)
                .limit(2)
                .stream()
            )
        if len(candidates) != 1:
            raise HTTPException(
                status_code=403,
                detail="此 Firebase 帳號尚未建立唯一的 Coaches 權限資料",
            )
        data = candidates[0].to_dict() or {}

    if data.get("disabled") is True:
        raise HTTPException(status_code=403, detail="此帳號已停用")

    role = _clean_id(data.get("role")).lower()
    if role not in {"coach", "manager", "owner"}:
        raise HTTPException(status_code=403, detail=f"不支援的 role：{role or '(empty)'}")

    company_id = _clean_id(data.get("company_id"))
    branch_id = _clean_id(data.get("branch_id")) or None
    coach_id = _clean_id(data.get("coach_id"))
    roles = _normalize_roles(data)

    if not company_id:
        raise HTTPException(status_code=403, detail="Coaches 缺少 company_id")
    if not coach_id:
        raise HTTPException(status_code=403, detail="Coaches 缺少 coach_id")
    if role in {"coach", "manager"} and not branch_id:
        raise HTTPException(status_code=403, detail=f"role={role} 必須設定 branch_id")
    if "coach" not in roles:
        raise HTTPException(status_code=403, detail="此帳號沒有 coach 上傳屬性")

    return ScannerActor(
        firebase_uid=uid,
        company_id=company_id,
        branch_id=branch_id,
        coach_id=coach_id,
        role=role,
        roles=roles,
    )


def require_scanner_actor(
    authorization: str | None = Header(default=None),
) -> ScannerActor:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="缺少 Firebase Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Firebase token 為空")

    try:
        decoded = firebase_auth.verify_id_token(token, clock_skew_seconds=5)
    except Exception as exc:
        logger.warning("Scanner Firebase token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="登入已失效，請重新登入") from exc

    return _resolve_scanner_actor(decoded)


def _branch_belongs_to_company(branch_id: str, company_id: str) -> bool:
    branch_id = _clean_id(branch_id)
    company_id = _clean_id(company_id)
    if not branch_id or not company_id:
        return False

    snap = app.state.db.collection("Branches").document(branch_id).get()
    if not snap.exists:
        return False

    return _clean_id((snap.to_dict() or {}).get("company_id")) == company_id


def _resolve_upload_branch(actor: ScannerActor, requested_branch_id: str) -> str:
    branch_id = _clean_id(requested_branch_id)
    if not branch_id:
        raise HTTPException(status_code=422, detail="branch_id 不可為空")

    if actor.role in {"coach", "manager"}:
        if branch_id != actor.branch_id:
            raise HTTPException(status_code=403, detail="無權在其他分店上傳資料")
        return branch_id

    # owner chooses branch at scan time.
    if not _branch_belongs_to_company(branch_id, actor.company_id):
        raise HTTPException(status_code=403, detail="所選 branch 不屬於目前 owner 的企業")
    return branch_id


def _validate_scanner_context(
    actor: ScannerActor,
    company_id: str,
    branch_id: str,
    coach_id: str,
    used_id: str,
) -> tuple[str, str, str, dict]:
    """
    Upload uses coach property; read permission remains a separate Coach API role concern.
    Scanner also requires an exact Students association for this coach + member.
    """
    requested_company = _clean_id(company_id)
    requested_coach = _clean_id(coach_id)
    requested_user = _clean_id(used_id)

    if requested_company != actor.company_id:
        raise HTTPException(status_code=403, detail="company_id 與登入帳號不一致")
    if requested_coach != actor.coach_id:
        raise HTTPException(status_code=403, detail="coach_id 與登入帳號的教練屬性不一致")
    if not requested_user:
        raise HTTPException(status_code=422, detail="target_user_id 不可為空")

    resolved_branch = _resolve_upload_branch(actor, branch_id)

    matches = list(
        app.state.db.collection("Students")
        .where("company_id", "==", actor.company_id)
        .where("branch_id", "==", resolved_branch)
        .where("coach_id", "==", actor.coach_id)
        .where("used_id", "==", requested_user)
        .limit(2)
        .stream()
    )
    if not matches:
        raise HTTPException(
            status_code=403,
            detail="此學員未與目前 coach_id 在所選 branch 建立 Students 關聯",
        )

    student_data = matches[0].to_dict() or {}
    return actor.company_id, resolved_branch, actor.coach_id, student_data


# -----------------------------------------------------------------------------
# 2. 圖片前處理
# -----------------------------------------------------------------------------
def preprocess_image(image: Image.Image, max_size=(1024, 1024)) -> Image.Image:
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    return image


# -----------------------------------------------------------------------------
# 3. Gemini 報表擷取：使用 Structured Output 強制 JSON Schema
# -----------------------------------------------------------------------------
class BodyCompositionExtraction(BaseModel):
    """Gemini OCR 的固定輸出 schema。所有讀不到的欄位皆允許為 null。"""

    device_type: Literal["inbody", "tanita", "other"] = "other"
    age: int | None = None
    height_m: float | None = None
    weight_kg: float | None = None
    body_fat_percentage: float | None = None
    body_fat_mass_kg: float | None = None
    fat_free_mass_kg: float | None = None
    basal_metabolic_rate: float | None = None
    smi_kg_m2: float | None = None
    visceral_fat_value: float | None = None
    visceral_fat_reference_low: float | None = None
    visceral_fat_reference_high: float | None = None
    inbody_score: float | None = None
    gender: Literal["male", "female"] | None = None
    segmental_muscle_ra: float | None = None
    segmental_muscle_la: float | None = None
    segmental_muscle_rl: float | None = None
    segmental_muscle_ll: float | None = None


def extract_health_data(image: Image.Image, metric_context: dict[str, Any] | None = None) -> dict:
    client = app.state.genai_client

    prompt = r"""
你是一個專業 BIA 身體組成分析報表資料擷取器。
任務是從 InBody、TANITA 或其他 BIA 報表擷取身體組成資料。

請只根據報表上實際顯示的內容擷取資料；看不清楚、沒有顯示、無法確定時填 null。
不要猜測、不要自行補值、不要自行計算報表沒有直接提供的欄位。

欄位定義：
- device_type：品牌，inbody / tanita / other。
- age：報表直接顯示的年齡；沒有顯示或無法確定時填 null。禁止由外觀推測。
- height_m：身高，統一轉公尺，例如 170 cm => 1.70。
- weight_kg：體重 kg。
- body_fat_percentage：Percent Body Fat / PBF / 體脂肪率 (%)。
- body_fat_mass_kg：Body Fat Mass / Fat Mass，kg。
- fat_free_mass_kg：Fat Free Mass / FFM，kg；禁止以 Muscle Mass 代替。
- basal_metabolic_rate：BMR，kcal/day。
- smi_kg_m2：報表直接顯示的 SMI，kg/m²；沒有直接顯示就 null。
- visceral_fat_value：報表上的 Visceral Fat / 內臟脂肪數值。不要使用 Visceral Fat Area (cm²)。
- visceral_fat_reference_low：
  - InBody：一律填 null。不要把 Visceral Fat Level 圖上的 Low/High 標籤或 <10 讀成 reference low/high。
  - TANITA / other：只有報表明確提供與 visceral_fat_value 同量尺的正常/理想範圍下限才擷取，否則 null。
- visceral_fat_reference_high：
  - InBody：一律填 null。不要把 Visceral Fat Level 圖上的 Low/High 標籤或 <10 讀成 reference low/high。
  - TANITA / other：只有報表明確提供與 visceral_fat_value 同量尺的正常/理想範圍上限才擷取，否則 null。
- inbody_score：只有 InBody 報表明確顯示 InBody Score 才填；TANITA/other 填 null。
- gender：只讀取報表直接顯示的 Sex / Gender / 性別。男性 => male，女性 => female；看不到就 null。禁止由姓名、外觀、年齡推測。
- segmental_muscle_ra：右臂肌肉量 kg。
- segmental_muscle_la：左臂肌肉量 kg。
- segmental_muscle_rl：右腿肌肉量 kg。
- segmental_muscle_ll：左腿肌肉量 kg。

重要規則：
- 所有數值都必須來自報表實際資訊；僅允許單位換算（例如 cm -> m）。
- 不要擷取 Visceral Fat Area (cm²)。
- 不要把 Body Fat Percentage 和 Visceral Fat 混在一起。
- 不要自行計算 ALM、SMI、BMI、AI；這些由 Python 後端計算。
"""

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG", quality=90)
    image_bytes = img_byte_arr.getvalue()

    response = _generate_content_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BodyCompositionExtraction,
            temperature=0.0,
        ),
        operation="inbody_recognition",
        metric_context={
            **(metric_context or {}),
            "prompt_version": "bodycomp_structured_v1_20260929",
            "schema_output_version": "BodyCompositionExtraction_v1",
            "image_count": 1,
            "image_bytes": len(image_bytes),
        },
    )

    try:
        # google-genai 在 response_schema 模式下通常會提供 parsed。
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BodyCompositionExtraction):
            result = parsed.model_dump()
        elif isinstance(parsed, dict):
            result = BodyCompositionExtraction.model_validate(parsed).model_dump()
        else:
            # 保留文字解析 fallback；Pydantic 會再次驗證 schema。
            raw_text = (response.text or "").strip()
            result = BodyCompositionExtraction.model_validate_json(raw_text).model_dump()

        logger.info(
            f"🤖 Gemini 解析完成，model={GEMINI_MODEL}, structured_output=True"
        )
        return result

    except Exception as exc:
        raw_text = ""
        try:
            raw_text = (response.text or "").strip()
        except Exception:
            pass
        logger.error(
            f"AI Structured Output 內容解析失敗: {exc}; "
            f"raw_response={raw_text[:2000]!r}"
        )
        raise HTTPException(
            status_code=500,
            detail="AI 已回應，但結構化資料解析失敗",
        ) from exc



# -----------------------------------------------------------------------------
# 3A. Muzili 專屬雙圖身體組成擷取
# -----------------------------------------------------------------------------
def extract_muzili_health_data(
    body_data_image: Image.Image,
    segmental_muscle_image: Image.Image,
    metric_context: dict[str, Any] | None = None,
) -> dict:
    """Muzili 專屬雙圖辨識。

    圖片 1：Muzili「身體數據」
    圖片 2：Muzili「節段分析 -> 肌肉量」

    height / age / gender 不由 Gemini 判斷，會由 API endpoint 的人工輸入覆蓋。
    """

    client = app.state.genai_client

    prompt = r"""
你是一個 CoachOS Muzili 身體組成資料擷取器。

本次固定會提供兩張 Muzili App 截圖，圖片順序固定：

【圖片 1：身體數據】
只從此圖片擷取：
- weight_kg：體重 kg
- body_fat_percentage：體脂率 %
- body_fat_mass_kg：脂肪量 kg
- fat_free_mass_kg：去脂體重 kg
- basal_metabolic_rate：基礎代謝 kcal/day
- visceral_fat_value：內臟脂肪數值

【圖片 2：節段分析 -> 肌肉量】
只從此圖片擷取：
- segmental_muscle_la：左臂肌肉量 kg
- segmental_muscle_ra：右臂肌肉量 kg
- segmental_muscle_ll：左腿肌肉量 kg
- segmental_muscle_rl：右腿肌肉量 kg

固定輸出規則：
- device_type 一律為 "other"。
- age 一律填 null。
- height_m 一律填 null。
- gender 一律填 null。
- smi_kg_m2 一律填 null。
- inbody_score 一律填 null。
- visceral_fat_reference_low 一律填 null。
- visceral_fat_reference_high 一律填 null。

非常重要：
- Muzili 的「身體年齡」不是實際年齡，不可擷取成 age。
- FFMI 不是 SMI，禁止把 FFMI 填入 smi_kg_m2。
- 人物模型的外觀不能用來判斷 gender。
- 「肌肉量」與「去脂體重」不同，不可互相替代。
- 圖片 2 若是「脂肪量」頁面而不是「肌肉量」頁面，不得把脂肪數值填入 segmental_muscle 欄位。
- 看不清楚或沒有顯示的欄位填 null。
- 不自行猜測或補值。
"""

    def image_to_jpeg_bytes(image: Image.Image) -> bytes:
        img = image.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue()

    body_bytes = image_to_jpeg_bytes(body_data_image)
    segmental_bytes = image_to_jpeg_bytes(segmental_muscle_image)

    response = _generate_content_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=body_bytes, mime_type="image/jpeg"),
            types.Part.from_bytes(data=segmental_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BodyCompositionExtraction,
            temperature=0.0,
        ),
        # 沿用既有 SBIR operation 名稱，避免破壞目前三類 AIUsageMetrics 統計。
        operation="inbody_recognition",
        metric_context={
            **(metric_context or {}),
            "source_system": "muzili",
            "prompt_version": "muzili_bodycomp_v1_20261006",
            "schema_output_version": "BodyCompositionExtraction_v1",
            "image_count": 2,
            "image_bytes": len(body_bytes) + len(segmental_bytes),
        },
    )

    try:
        parsed = getattr(response, "parsed", None)

        if isinstance(parsed, BodyCompositionExtraction):
            result = parsed.model_dump()
        elif isinstance(parsed, dict):
            result = BodyCompositionExtraction.model_validate(parsed).model_dump()
        else:
            raw_text = (response.text or "").strip()
            result = BodyCompositionExtraction.model_validate_json(raw_text).model_dump()

        # 防呆：這些欄位不採信 AI，後續由人工欄位 / Python deterministic logic 決定。
        result["device_type"] = "other"
        result["age"] = None
        result["height_m"] = None
        result["gender"] = None
        result["smi_kg_m2"] = None
        result["inbody_score"] = None
        result["visceral_fat_reference_low"] = None
        result["visceral_fat_reference_high"] = None

        logger.info(
            "Muzili 雙圖辨識完成 model=%s structured_output=True",
            GEMINI_MODEL,
        )
        return result

    except Exception as exc:
        raw_text = ""
        try:
            raw_text = (response.text or "").strip()
        except Exception:
            pass

        logger.error(
            "Muzili Structured Output 解析失敗: %s; raw_response=%r",
            exc,
            raw_text[:2000],
        )
        raise HTTPException(
            status_code=500,
            detail="Muzili AI 已回應，但結構化資料解析失敗",
        ) from exc


# -----------------------------------------------------------------------------
# 3B. CoachOS 標準問卷擷取：固定版型 -> Boolean canonical schema
# -----------------------------------------------------------------------------
QUESTIONNAIRE_VERSION = "QST_v2.1"
HISTORY_VERSION = "HIS_v2.1"

# Scanner API 責任僅限：
# CoachOS 標準問卷圖片 -> Gemini 固定版型辨識 -> canonical boolean schema -> Firestore。
# Rule Base、Cloud Function、運動建議與溝通生成由其他服務負責。

HISTORY_QUESTIONS = [{'id': 'H01',
  'section': 'history',
  'question': '目前是否有規律運動？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'exercises_regularly_currently', 'value': True},
              {'label': '否', 'key': 'exercises_regularly_currently', 'value': False}]},
 {'id': 'H02',
  'section': 'history',
  'question': '目前是否有醫師交代的運動限制？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'physician_exercise_restriction', 'value': True},
              {'label': '否', 'key': 'physician_exercise_restriction', 'value': False}]},
 {'id': 'H03',
  'section': 'history',
  'question': '休息、日常活動或運動時，是否曾有胸痛？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'chest_pain_at_rest_or_activity', 'value': True},
              {'label': '否', 'key': 'chest_pain_at_rest_or_activity', 'value': False}]},
 {'id': 'H04',
  'section': 'history',
  'question': '是否有以下病史或健康狀況？',
  'type': 'multiple',
  'options': [{'label': '無', 'control_key': 'h04_none_selected', 'value': True},
              {'label': '高血壓', 'key': 'hypertension', 'value': True},
              {'label': '低血壓', 'key': 'hypotension', 'value': True},
              {'label': '心臟相關疾病', 'key': 'heart_related_condition', 'value': True},
              {'label': '低血糖症狀', 'key': 'hypoglycemia_symptom', 'value': True},
              {'label': '腎臟相關疾病', 'key': 'kidney_related_condition', 'value': True},
              {'label': '肝臟相關疾病', 'key': 'liver_related_condition', 'value': True},
              {'label': '肺臟相關疾病', 'key': 'lung_related_condition', 'value': True},
              {'label': '骨質疏鬆', 'key': 'osteoporosis', 'value': True},
              {'label': '巴金森氏症', 'key': 'parkinsons_disease', 'value': True},
              {'label': '神經相關疾病', 'key': 'neurological_condition', 'value': True}]},
 {'id': 'H05',
  'section': 'history',
  'question': '目前身體哪些部位有不適（包含疼痛）？',
  'type': 'multiple',
  'options': [{'label': '無', 'control_key': 'h05_none_selected', 'value': True},
              {'label': '頸部', 'key': 'discomfort_neck', 'value': True},
              {'label': '肩部', 'key': 'discomfort_shoulder', 'value': True},
              {'label': '手肘', 'key': 'discomfort_elbow', 'value': True},
              {'label': '手腕／手部', 'key': 'discomfort_wrist_hand', 'value': True},
              {'label': '上背', 'key': 'discomfort_upper_back', 'value': True},
              {'label': '下背', 'key': 'discomfort_low_back', 'value': True},
              {'label': '髖部', 'key': 'discomfort_hip', 'value': True},
              {'label': '膝部', 'key': 'discomfort_knee', 'value': True},
              {'label': '踝部／足部', 'key': 'discomfort_ankle_foot', 'value': True}]},
 {'id': 'H06',
  'section': 'history',
  'question': '過去 12 個月內是否曾跌倒？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'fall_last_12_months', 'value': True},
              {'label': '否', 'key': 'fall_last_12_months', 'value': False}]},
 {'id': 'H07',
  'section': 'history',
  'question': '是否會擔心或害怕跌倒？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'fear_of_falling', 'value': True},
              {'label': '否', 'key': 'fear_of_falling', 'value': False}]},
 {'id': 'H08',
  'section': 'history',
  'question': '日常活動或運動時，是否曾有下腹部肌肉控制問題（這題由教練解釋）？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'urinary_leakage', 'value': True},
              {'label': '否', 'key': 'urinary_leakage', 'value': False}]},
 {'id': 'H09',
  'section': 'history',
  'question': '哪些部位曾進行過手術？',
  'type': 'multiple',
  'options': [{'label': '無', 'control_key': 'h09_none_selected', 'value': True},
              {'label': '頸部', 'key': 'surgery_cervical_spine', 'value': True},
              {'label': '肩部', 'key': 'surgery_shoulder', 'value': True},
              {'label': '手肘', 'key': 'surgery_elbow', 'value': True},
              {'label': '手腕／手部', 'key': 'surgery_wrist_hand', 'value': True},
              {'label': '上背', 'key': 'surgery_thoracic_spine', 'value': True},
              {'label': '下背', 'key': 'surgery_lumbar_spine', 'value': True},
              {'label': '髖部', 'key': 'surgery_hip', 'value': True},
              {'label': '膝部', 'key': 'surgery_knee', 'value': True},
              {'label': '踝部／足部', 'key': 'surgery_ankle_foot', 'value': True}]},
 {'id': 'H10',
  'section': 'history',
  'question': '最近一次手術距今多久？',
  'type': 'single',
  'options': [{'label': '沒有手術史', 'key': 'no_surgery_history', 'value': True},
              {'label': '3 個月內', 'key': 'surgery_most_recent_within_3_months', 'value': True},
              {'label': '3–6 個月', 'key': 'surgery_most_recent_3_6_months', 'value': True},
              {'label': '6–12 個月', 'key': 'surgery_most_recent_6_12_months', 'value': True},
              {'label': '超過 1 年', 'key': 'surgery_most_recent_over_1_year', 'value': True}]}]

QUESTIONNAIRE_QUESTIONS = [{'id': 'Q01',
  'section': 'questionnaire',
  'question': '目前每週運動頻率？',
  'type': 'single',
  'options': [{'label': '幾乎沒有規律運動', 'key': 'exercise_frequency_none', 'value': True},
              {'label': '每週 1–2 次', 'key': 'exercise_frequency_1_2_per_week', 'value': True},
              {'label': '每週 3–4 次', 'key': 'exercise_frequency_3_4_per_week', 'value': True},
              {'label': '每週 5 次以上', 'key': 'exercise_frequency_5_plus_per_week', 'value': True}]},
 {'id': 'Q02',
  'section': 'questionnaire',
  'question': '每次運動通常多久？',
  'type': 'single',
  'options': [{'label': '少於 30 分鐘', 'key': 'session_duration_under_30_min', 'value': True},
              {'label': '30–60 分鐘', 'key': 'session_duration_30_60_min', 'value': True},
              {'label': '超過 60 分鐘', 'key': 'session_duration_over_60_min', 'value': True}]},
 {'id': 'Q03',
  'section': 'questionnaire',
  'question': '維持目前運動習慣多久？',
  'type': 'single',
  'options': [{'label': '少於 3 個月', 'key': 'exercise_history_under_3_months', 'value': True},
              {'label': '3–12 個月', 'key': 'exercise_history_3_12_months', 'value': True},
              {'label': '超過 1 年', 'key': 'exercise_history_over_1_year', 'value': True}]},
 {'id': 'Q04',
  'section': 'questionnaire',
  'question': '每天久坐時間大約多久？',
  'type': 'single',
  'options': [{'label': '少於 4 小時', 'key': 'sedentary_time_under_4_hours', 'value': True},
              {'label': '4–8 小時', 'key': 'sedentary_time_4_8_hours', 'value': True},
              {'label': '超過 8 小時', 'key': 'sedentary_time_over_8_hours', 'value': True}]},
 {'id': 'Q05',
  'section': 'questionnaire',
  'question': '偏好的運動強度？',
  'type': 'single',
  'options': [{'label': '輕鬆強度', 'key': 'prefers_light_intensity', 'value': True},
              {'label': '中等強度', 'key': 'prefers_moderate_intensity', 'value': True},
              {'label': '較有挑戰的高強度', 'key': 'prefers_vigorous_intensity', 'value': True}]},
 {'id': 'Q06',
  'section': 'questionnaire',
  'question': '喜歡哪些運動類型？',
  'type': 'multiple',
  'options': [{'label': '散步', 'key': 'likes_walking', 'value': True},
              {'label': '健走', 'key': 'likes_brisk_walking', 'value': True},
              {'label': '慢跑', 'key': 'likes_jogging', 'value': True},
              {'label': '跑步', 'key': 'likes_running', 'value': True},
              {'label': '馬拉松／路跑', 'key': 'likes_marathon_or_road_race', 'value': True},
              {'label': '登山', 'key': 'likes_hiking', 'value': True},
              {'label': '步道／戶外越野', 'key': 'likes_trail_activity', 'value': True},
              {'label': '自行車', 'key': 'likes_cycling', 'value': True},
              {'label': '水中運動', 'key': 'likes_aquatic_exercise', 'value': True},
              {'label': '重量／阻力訓練', 'key': 'likes_resistance_training', 'value': True},
              {'label': '瑜伽', 'key': 'likes_yoga', 'value': True},
              {'label': '皮拉提斯', 'key': 'likes_pilates', 'value': True},
              {'label': '太極／氣功', 'key': 'likes_tai_chi_or_qigong', 'value': True},
              {'label': '舞蹈', 'key': 'likes_dance', 'value': True},
              {'label': '羽球／網球／桌球等拍類', 'key': 'likes_racket_sports', 'value': True},
              {'label': '球類運動', 'key': 'likes_ball_sports', 'value': True},
              {'label': '高爾夫', 'key': 'likes_golf', 'value': True},
              {'label': '戶外運動／戶外健身', 'key': 'likes_outdoor_fitness', 'value': True},
              {'label': '伸展／活動度／放鬆', 'key': 'likes_stretching_mobility', 'value': True}]},
 {'id': 'Q07',
  'section': 'questionnaire',
  'question': '偏好哪些運動場域？',
  'type': 'multiple',
  'options': [{'label': '室內', 'key': 'prefers_indoor', 'value': True},
              {'label': '室外', 'key': 'prefers_outdoor', 'value': True},
              {'label': '在家', 'key': 'prefers_home', 'value': True},
              {'label': '健身房', 'key': 'prefers_gym', 'value': True},
              {'label': '社區／活動中心', 'key': 'prefers_community_center', 'value': True},
              {'label': '公園／戶外場地', 'key': 'prefers_park', 'value': True}]},
 {'id': 'Q08',
  'section': 'questionnaire',
  'question': '偏好哪種運動互動方式？',
  'type': 'multiple',
  'options': [{'label': '自己運動', 'key': 'prefers_exercise_alone', 'value': True},
              {'label': '一對一陪同', 'key': 'prefers_one_on_one_coaching', 'value': True},
              {'label': '小團體', 'key': 'prefers_small_group', 'value': True},
              {'label': '多人團課', 'key': 'prefers_large_group_class', 'value': True},
              {'label': '與家人朋友一起活動', 'key': 'prefers_with_family_or_friends', 'value': True}]},
 {'id': 'Q09',
  'section': 'questionnaire',
  'question': '希望透過運動達成哪些目標？',
  'type': 'multiple',
  'options': [{'label': '維持／提升整體健康', 'key': 'goal_general_health', 'value': True},
              {'label': '控制體重／體脂', 'key': 'goal_weight_management', 'value': True},
              {'label': '改善體態', 'key': 'goal_body_shape', 'value': True},
              {'label': '增加肌力或肌肉量', 'key': 'goal_strength_or_muscle_gain', 'value': True},
              {'label': '提升柔軟度／關節活動度', 'key': 'goal_mobility', 'value': True},
              {'label': '提升平衡', 'key': 'goal_balance', 'value': True},
              {'label': '降低跌倒風險', 'key': 'goal_fall_prevention', 'value': True},
              {'label': '維持獨立生活能力', 'key': 'goal_independence', 'value': True},
              {'label': '為旅遊／走路／行程體力做準備', 'key': 'goal_travel_fitness', 'value': True},
              {'label': '改善睡眠／壓力', 'key': 'goal_sleep_stress', 'value': True},
              {'label': '增加社交', 'key': 'goal_social_connection', 'value': True},
              {'label': '建立規律運動習慣', 'key': 'goal_build_exercise_habit', 'value': True},
              {'label': '更有精神與活力', 'key': 'goal_energy_vitality', 'value': True}]},
 {'id': 'Q10',
  'section': 'questionnaire',
  'question': '哪些因素可能影響你持續運動？',
  'type': 'multiple',
  'options': [{'label': '疼痛', 'key': 'barrier_pain', 'value': True},
              {'label': '容易疲勞', 'key': 'barrier_fatigue', 'value': True},
              {'label': '怕受傷', 'key': 'barrier_fear_of_injury', 'value': True},
              {'label': '對運動沒信心', 'key': 'barrier_low_confidence', 'value': True},
              {'label': '時間安排', 'key': 'barrier_time', 'value': True},
              {'label': '交通', 'key': 'barrier_transportation', 'value': True},
              {'label': '費用', 'key': 'barrier_cost', 'value': True},
              {'label': '天氣', 'key': 'barrier_weather', 'value': True},
              {'label': '沒有同伴', 'key': 'barrier_no_companion', 'value': True},
              {'label': '不喜歡團體活動', 'key': 'barrier_dislikes_group', 'value': True}]},
 {'id': 'Q11',
  'section': 'questionnaire',
  'question': '偏好教練如何說明與互動？',
  'type': 'multiple',
  'options': [{'label': '一步一步清楚說明', 'key': 'communication_prefers_step_by_step', 'value': True},
              {'label': '先看示範再做', 'key': 'communication_prefers_demonstration', 'value': True},
              {'label': '看數據與進步幅度', 'key': 'communication_prefers_data_progress', 'value': True},
              {'label': '鼓勵與正向回饋', 'key': 'communication_prefers_encouragement', 'value': True},
              {'label': '有挑戰與目標感', 'key': 'communication_prefers_challenge', 'value': True},
              {'label': '輕鬆聊天式互動', 'key': 'communication_prefers_casual_chat', 'value': True},
              {'label': '安靜專注、少聊天', 'key': 'communication_prefers_quiet_focus', 'value': True},
              {'label': '每次有些變化', 'key': 'communication_prefers_variety', 'value': True},
              {'label': '固定熟悉的流程', 'key': 'communication_prefers_routine', 'value': True},
              {'label': '較慢速度、重點清楚的說明',
               'key': 'communication_prefers_slow_pace_explanation',
               'value': True}]},
 {'id': 'Q12',
  'section': 'questionnaire',
  'question': '哪些事情最能激勵你持續運動？',
  'type': 'multiple',
  'options': [{'label': '健康改善', 'key': 'motivation_health', 'value': True},
              {'label': '家人支持', 'key': 'motivation_family', 'value': True},
              {'label': '旅遊與行動力', 'key': 'motivation_travel', 'value': True},
              {'label': '外觀／體態變化', 'key': 'motivation_appearance', 'value': True},
              {'label': '運動表現／挑戰目標', 'key': 'motivation_performance', 'value': True},
              {'label': '社交與同伴', 'key': 'motivation_social', 'value': True},
              {'label': '專業建議', 'key': 'motivation_professional_advice', 'value': True},
              {'label': '看到可量化進步', 'key': 'motivation_measurable_progress', 'value': True}]},
 {'id': 'Q13',
  'section': 'questionnaire',
  'question': '哪些時段較方便運動？',
  'type': 'multiple',
  'options': [{'label': '早上', 'key': 'available_morning', 'value': True},
              {'label': '下午', 'key': 'available_afternoon', 'value': True},
              {'label': '晚上', 'key': 'available_evening', 'value': True},
              {'label': '平日', 'key': 'available_weekday', 'value': True},
              {'label': '週末', 'key': 'available_weekend', 'value': True}]},
 {'id': 'Q14',
  'section': 'questionnaire',
  'question': '是否喜歡旅遊？',
  'type': 'single',
  'options': [{'label': '是', 'key': 'likes_travel', 'value': True},
              {'label': '否', 'key': 'likes_travel', 'value': False}]},
 {'id': 'Q15',
  'section': 'questionnaire',
  'question': '近一年旅遊頻率？',
  'type': 'single',
  'options': [{'label': '沒有旅遊', 'key': 'travel_frequency_none', 'value': True},
              {'label': '約 1 次', 'key': 'travel_frequency_1_per_year', 'value': True},
              {'label': '2–3 次', 'key': 'travel_frequency_2_3_per_year', 'value': True},
              {'label': '4 次以上', 'key': 'travel_frequency_4_plus_per_year', 'value': True}]},
 {'id': 'Q16',
  'section': 'questionnaire',
  'question': '旅遊範圍以哪一類為主？',
  'type': 'single',
  'options': [{'label': '國內為主', 'key': 'travel_scope_domestic_more', 'value': True},
              {'label': '國外為主', 'key': 'travel_scope_international_more', 'value': True},
              {'label': '國內與國外都常安排', 'key': 'travel_scope_both', 'value': True}]},
 {'id': 'Q17',
  'section': 'questionnaire',
  'question': '最近一次旅遊距今多久？',
  'type': 'single',
  'options': [{'label': '3 個月內', 'key': 'last_trip_within_3_months', 'value': True},
              {'label': '3–6 個月', 'key': 'last_trip_3_6_months', 'value': True},
              {'label': '6–12 個月', 'key': 'last_trip_6_12_months', 'value': True},
              {'label': '超過 1 年', 'key': 'last_trip_over_1_year', 'value': True}]},
 {'id': 'Q18',
  'section': 'questionnaire',
  'question': '常去哪些旅遊地區？',
  'type': 'multiple',
  'options': [{'label': '台灣國內', 'key': 'travel_destination_domestic', 'value': True},
              {'label': '日本', 'key': 'travel_destination_japan', 'value': True},
              {'label': '韓國', 'key': 'travel_destination_korea', 'value': True},
              {'label': '東南亞', 'key': 'travel_destination_southeast_asia', 'value': True},
              {'label': '中國大陸／香港／澳門', 'key': 'travel_destination_greater_china_hk_macau', 'value': True},
              {'label': '歐洲', 'key': 'travel_destination_europe', 'value': True},
              {'label': '北美', 'key': 'travel_destination_north_america', 'value': True},
              {'label': '澳洲／紐西蘭', 'key': 'travel_destination_oceania', 'value': True},
              {'label': '其他國外地區', 'key': 'travel_destination_other_international', 'value': True}]},
 {'id': 'Q19',
  'section': 'questionnaire',
  'question': '偏好哪些旅遊方式？',
  'type': 'multiple',
  'options': [{'label': '自由行', 'key': 'travel_style_independent', 'value': True},
              {'label': '跟團', 'key': 'travel_style_group_tour', 'value': True},
              {'label': '郵輪', 'key': 'travel_style_cruise', 'value': True},
              {'label': '自駕／公路旅行', 'key': 'travel_style_road_trip', 'value': True},
              {'label': '健行／滑雪／單車等活動型', 'key': 'travel_style_active', 'value': True},
              {'label': '美食／文化／城市型', 'key': 'travel_style_food_culture', 'value': True},
              {'label': '自然景觀型', 'key': 'travel_style_nature', 'value': True}]},
 {'id': 'Q20',
  'section': 'questionnaire',
  'question': '平常喜歡哪些休閒活動？',
  'type': 'multiple',
  'options': [{'label': '戶外活動', 'key': 'leisure_outdoor_activity', 'value': True},
              {'label': '山林／自然景點', 'key': 'leisure_hiking_nature', 'value': True},
              {'label': '園藝', 'key': 'leisure_gardening', 'value': True},
              {'label': '攝影', 'key': 'leisure_photography', 'value': True},
              {'label': '閱讀', 'key': 'leisure_reading', 'value': True},
              {'label': '音樂', 'key': 'leisure_music', 'value': True},
              {'label': '唱歌', 'key': 'leisure_singing', 'value': True},
              {'label': '跳舞', 'key': 'leisure_dancing', 'value': True},
              {'label': '料理', 'key': 'leisure_cooking', 'value': True},
              {'label': '美食／咖啡館', 'key': 'leisure_food_cafe', 'value': True},
              {'label': '電影／影視', 'key': 'leisure_movies_tv', 'value': True},
              {'label': '展覽／藝文／文化活動', 'key': 'leisure_arts_culture', 'value': True},
              {'label': '逛街購物', 'key': 'leisure_shopping', 'value': True},
              {'label': '志工活動', 'key': 'leisure_volunteering', 'value': True},
              {'label': '社團／社區活動', 'key': 'leisure_community_club', 'value': True},
              {'label': '家庭活動', 'key': 'leisure_family_time', 'value': True},
              {'label': '寵物相關活動', 'key': 'leisure_pets', 'value': True},
              {'label': '桌遊／棋牌／益智活動', 'key': 'leisure_board_games_cards', 'value': True}]},
 {'id': 'Q21',
  'section': 'questionnaire',
  'question': '哪些描述比較像你？',
  'type': 'multiple',
  'options': [{'label': '樂觀陽光', 'key': 'self_description_optimistic', 'value': True},
              {'label': '外向', 'key': 'self_description_outgoing', 'value': True},
              {'label': '沉穩', 'key': 'self_description_calm', 'value': True},
              {'label': '做事謹慎', 'key': 'self_description_cautious', 'value': True},
              {'label': '喜歡競賽／挑戰', 'key': 'self_description_competitive', 'value': True},
              {'label': '喜歡自主安排', 'key': 'self_description_independent', 'value': True},
              {'label': '喜歡與人互動', 'key': 'self_description_social', 'value': True},
              {'label': '喜歡嘗試新事物', 'key': 'self_description_curious', 'value': True},
              {'label': '喜歡固定熟悉的節奏', 'key': 'self_description_routine_oriented', 'value': True},
              {'label': '喜歡冒險／探索', 'key': 'self_description_adventurous', 'value': True},
              {'label': '有耐心', 'key': 'self_description_patient', 'value': True}]},
 {'id': 'Q22',
  'section': 'questionnaire',
  'question': '偏好哪些資訊呈現或數位使用方式？',
  'type': 'multiple',
  'options': [{'label': '使用智慧型手機很自在', 'key': 'comfortable_using_smartphone', 'value': True},
              {'label': '喜歡用 App 看進度／紀錄', 'key': 'likes_app_progress_tracking', 'value': True},
              {'label': '偏好紙本資訊', 'key': 'prefers_printed_information', 'value': True},
              {'label': '偏好較大文字與清楚版面', 'key': 'prefers_large_text', 'value': True}]}]

HISTORY_FIELD_LABELS: dict[str, str] = {'exercises_regularly_currently': '目前是否有規律運動',
 'physician_exercise_restriction': '目前是否有醫師交代的運動限制',
 'chest_pain_at_rest_or_activity': '休息、日常活動或運動時，是否曾有胸痛',
 'hypertension': '高血壓',
 'hypotension': '低血壓',
 'heart_related_condition': '心臟相關疾病',
 'hypoglycemia_symptom': '低血糖症狀',
 'kidney_related_condition': '腎臟相關疾病',
 'liver_related_condition': '肝臟相關疾病',
 'lung_related_condition': '肺臟相關疾病',
 'osteoporosis': '骨質疏鬆',
 'parkinsons_disease': '巴金森氏症',
 'neurological_condition': '神經相關疾病',
 'discomfort_neck': '頸部不適',
 'discomfort_shoulder': '肩部不適',
 'discomfort_elbow': '手肘不適',
 'discomfort_wrist_hand': '手腕／手部不適',
 'discomfort_upper_back': '上背不適',
 'discomfort_low_back': '下背不適',
 'discomfort_hip': '髖部不適',
 'discomfort_knee': '膝部不適',
 'discomfort_ankle_foot': '踝部／足部不適',
 'fall_last_12_months': '過去 12 個月內是否曾跌倒',
 'fear_of_falling': '是否會擔心或害怕跌倒',
 'urinary_leakage': '日常活動或運動時，曾有下腹部肌肉控制問題（對應 urinary_leakage）',
 'surgery_cervical_spine': '頸部曾手術',
 'surgery_shoulder': '肩部曾手術',
 'surgery_elbow': '手肘曾手術',
 'surgery_wrist_hand': '手腕／手部曾手術',
 'surgery_thoracic_spine': '上背曾手術',
 'surgery_lumbar_spine': '下背曾手術',
 'surgery_hip': '髖部曾手術',
 'surgery_knee': '膝部曾手術',
 'surgery_ankle_foot': '踝部／足部曾手術',
 'no_surgery_history': '最近一次手術：沒有手術史',
 'surgery_most_recent_within_3_months': '最近一次手術：3 個月內',
 'surgery_most_recent_3_6_months': '最近一次手術：3–6 個月',
 'surgery_most_recent_6_12_months': '最近一次手術：6–12 個月',
 'surgery_most_recent_over_1_year': '最近一次手術：超過 1 年'}
QUESTIONNAIRE_FIELD_LABELS: dict[str, str] = {'exercise_frequency_none': '幾乎沒有規律運動',
 'exercise_frequency_1_2_per_week': '每週 1–2 次',
 'exercise_frequency_3_4_per_week': '每週 3–4 次',
 'exercise_frequency_5_plus_per_week': '每週 5 次以上',
 'session_duration_under_30_min': '少於 30 分鐘',
 'session_duration_30_60_min': '30–60 分鐘',
 'session_duration_over_60_min': '超過 60 分鐘',
 'exercise_history_under_3_months': '少於 3 個月',
 'exercise_history_3_12_months': '3–12 個月',
 'exercise_history_over_1_year': '超過 1 年',
 'sedentary_time_under_4_hours': '少於 4 小時',
 'sedentary_time_4_8_hours': '4–8 小時',
 'sedentary_time_over_8_hours': '超過 8 小時',
 'prefers_light_intensity': '輕鬆強度',
 'prefers_moderate_intensity': '中等強度',
 'prefers_vigorous_intensity': '較有挑戰的高強度',
 'likes_walking': '散步',
 'likes_brisk_walking': '健走',
 'likes_jogging': '慢跑',
 'likes_running': '跑步',
 'likes_marathon_or_road_race': '馬拉松／路跑',
 'likes_hiking': '登山',
 'likes_trail_activity': '步道／戶外越野',
 'likes_cycling': '自行車',
 'likes_aquatic_exercise': '水中運動',
 'likes_resistance_training': '重量／阻力訓練',
 'likes_yoga': '瑜伽',
 'likes_pilates': '皮拉提斯',
 'likes_tai_chi_or_qigong': '太極／氣功',
 'likes_dance': '舞蹈',
 'likes_racket_sports': '羽球／網球／桌球等拍類',
 'likes_ball_sports': '球類運動',
 'likes_golf': '高爾夫',
 'likes_outdoor_fitness': '戶外運動／戶外健身',
 'likes_stretching_mobility': '伸展／活動度／放鬆',
 'prefers_indoor': '室內',
 'prefers_outdoor': '室外',
 'prefers_home': '在家',
 'prefers_gym': '健身房',
 'prefers_community_center': '社區／活動中心',
 'prefers_park': '公園／戶外場地',
 'prefers_exercise_alone': '自己運動',
 'prefers_one_on_one_coaching': '一對一陪同',
 'prefers_small_group': '小團體',
 'prefers_large_group_class': '多人團課',
 'prefers_with_family_or_friends': '與家人朋友一起活動',
 'goal_general_health': '維持／提升整體健康',
 'goal_weight_management': '控制體重／體脂',
 'goal_body_shape': '改善體態',
 'goal_strength_or_muscle_gain': '增加肌力或肌肉量',
 'goal_mobility': '提升柔軟度／關節活動度',
 'goal_balance': '提升平衡',
 'goal_fall_prevention': '降低跌倒風險',
 'goal_independence': '維持獨立生活能力',
 'goal_travel_fitness': '為旅遊／走路／行程體力做準備',
 'goal_sleep_stress': '改善睡眠／壓力',
 'goal_social_connection': '增加社交',
 'goal_build_exercise_habit': '建立規律運動習慣',
 'goal_energy_vitality': '更有精神與活力',
 'barrier_pain': '疼痛',
 'barrier_fatigue': '容易疲勞',
 'barrier_fear_of_injury': '怕受傷',
 'barrier_low_confidence': '對運動沒信心',
 'barrier_time': '時間安排',
 'barrier_transportation': '交通',
 'barrier_cost': '費用',
 'barrier_weather': '天氣',
 'barrier_no_companion': '沒有同伴',
 'barrier_dislikes_group': '不喜歡團體活動',
 'communication_prefers_step_by_step': '一步一步清楚說明',
 'communication_prefers_demonstration': '先看示範再做',
 'communication_prefers_data_progress': '看數據與進步幅度',
 'communication_prefers_encouragement': '鼓勵與正向回饋',
 'communication_prefers_challenge': '有挑戰與目標感',
 'communication_prefers_casual_chat': '輕鬆聊天式互動',
 'communication_prefers_quiet_focus': '安靜專注、少聊天',
 'communication_prefers_variety': '每次有些變化',
 'communication_prefers_routine': '固定熟悉的流程',
 'communication_prefers_slow_pace_explanation': '較慢速度、重點清楚的說明',
 'motivation_health': '健康改善',
 'motivation_family': '家人支持',
 'motivation_travel': '旅遊與行動力',
 'motivation_appearance': '外觀／體態變化',
 'motivation_performance': '運動表現／挑戰目標',
 'motivation_social': '社交與同伴',
 'motivation_professional_advice': '專業建議',
 'motivation_measurable_progress': '看到可量化進步',
 'available_morning': '早上',
 'available_afternoon': '下午',
 'available_evening': '晚上',
 'available_weekday': '平日',
 'available_weekend': '週末',
 'likes_travel': '是否喜歡旅遊',
 'travel_frequency_none': '沒有旅遊',
 'travel_frequency_1_per_year': '約 1 次',
 'travel_frequency_2_3_per_year': '2–3 次',
 'travel_frequency_4_plus_per_year': '4 次以上',
 'travel_scope_domestic_more': '國內為主',
 'travel_scope_international_more': '國外為主',
 'travel_scope_both': '國內與國外都常安排',
 'last_trip_within_3_months': '3 個月內',
 'last_trip_3_6_months': '3–6 個月',
 'last_trip_6_12_months': '6–12 個月',
 'last_trip_over_1_year': '超過 1 年',
 'travel_destination_domestic': '台灣國內',
 'travel_destination_japan': '日本',
 'travel_destination_korea': '韓國',
 'travel_destination_southeast_asia': '東南亞',
 'travel_destination_greater_china_hk_macau': '中國大陸／香港／澳門',
 'travel_destination_europe': '歐洲',
 'travel_destination_north_america': '北美',
 'travel_destination_oceania': '澳洲／紐西蘭',
 'travel_destination_other_international': '其他國外地區',
 'travel_style_independent': '自由行',
 'travel_style_group_tour': '跟團',
 'travel_style_cruise': '郵輪',
 'travel_style_road_trip': '自駕／公路旅行',
 'travel_style_active': '健行／滑雪／單車等活動型',
 'travel_style_food_culture': '美食／文化／城市型',
 'travel_style_nature': '自然景觀型',
 'leisure_outdoor_activity': '戶外活動',
 'leisure_hiking_nature': '山林／自然景點',
 'leisure_gardening': '園藝',
 'leisure_photography': '攝影',
 'leisure_reading': '閱讀',
 'leisure_music': '音樂',
 'leisure_singing': '唱歌',
 'leisure_dancing': '跳舞',
 'leisure_cooking': '料理',
 'leisure_food_cafe': '美食／咖啡館',
 'leisure_movies_tv': '電影／影視',
 'leisure_arts_culture': '展覽／藝文／文化活動',
 'leisure_shopping': '逛街購物',
 'leisure_volunteering': '志工活動',
 'leisure_community_club': '社團／社區活動',
 'leisure_family_time': '家庭活動',
 'leisure_pets': '寵物相關活動',
 'leisure_board_games_cards': '桌遊／棋牌／益智活動',
 'self_description_optimistic': '樂觀陽光',
 'self_description_outgoing': '外向',
 'self_description_calm': '沉穩',
 'self_description_cautious': '做事謹慎',
 'self_description_competitive': '喜歡競賽／挑戰',
 'self_description_independent': '喜歡自主安排',
 'self_description_social': '喜歡與人互動',
 'self_description_curious': '喜歡嘗試新事物',
 'self_description_routine_oriented': '喜歡固定熟悉的節奏',
 'self_description_adventurous': '喜歡冒險／探索',
 'self_description_patient': '有耐心',
 'comfortable_using_smartphone': '使用智慧型手機很自在',
 'likes_app_progress_tracking': '喜歡用 App 看進度／紀錄',
 'prefers_printed_information': '偏好紙本資訊',
 'prefers_large_text': '偏好較大文字與清楚版面'}

HISTORY_EXCLUSIVE_BOOL_GROUPS: dict[str, list[str]] = {'H10': ['no_surgery_history',
         'surgery_most_recent_within_3_months',
         'surgery_most_recent_3_6_months',
         'surgery_most_recent_6_12_months',
         'surgery_most_recent_over_1_year']}
QUESTIONNAIRE_EXCLUSIVE_BOOL_GROUPS: dict[str, list[str]] = {'Q01': ['exercise_frequency_none',
         'exercise_frequency_1_2_per_week',
         'exercise_frequency_3_4_per_week',
         'exercise_frequency_5_plus_per_week'],
 'Q02': ['session_duration_under_30_min', 'session_duration_30_60_min', 'session_duration_over_60_min'],
 'Q03': ['exercise_history_under_3_months', 'exercise_history_3_12_months', 'exercise_history_over_1_year'],
 'Q04': ['sedentary_time_under_4_hours', 'sedentary_time_4_8_hours', 'sedentary_time_over_8_hours'],
 'Q05': ['prefers_light_intensity', 'prefers_moderate_intensity', 'prefers_vigorous_intensity'],
 'Q15': ['travel_frequency_none',
         'travel_frequency_1_per_year',
         'travel_frequency_2_3_per_year',
         'travel_frequency_4_plus_per_year'],
 'Q16': ['travel_scope_domestic_more', 'travel_scope_international_more', 'travel_scope_both'],
 'Q17': ['last_trip_within_3_months',
         'last_trip_3_6_months',
         'last_trip_6_12_months',
         'last_trip_over_1_year']}



class QuestionnaireCompactBasicInfo(BaseModel):
    """只擷取問卷上清楚可見的基本資料；Firebase 綁定仍以 App 選取學員為準。"""
    name_on_form: str | None = None
    age: int | None = None
    gender: Literal["male", "female"] | None = None
    student_number_on_form: str | None = None


class QuestionnaireCompactAnswer(BaseModel):
    """
    Gemini 只回傳「題號 + 勾選文字」，避免把 192 個 Firebase boolean
    全部塞進 response_schema，降低 Gemini Structured Output 複雜度。
    """
    question_id: str
    visibility: Literal["complete", "partial", "unreadable"] = "complete"
    selected_labels: list[str] = Field(default_factory=list)
    unreadable_labels: list[str] = Field(default_factory=list)


class QuestionnaireCompactExtraction(BaseModel):
    basic_info: QuestionnaireCompactBasicInfo = Field(
        default_factory=QuestionnaireCompactBasicInfo
    )
    answers: list[QuestionnaireCompactAnswer] = Field(default_factory=list)
    unreadable_or_ambiguous_items: list[str] = Field(default_factory=list)


ALL_STANDARD_QUESTIONS: dict[str, dict] = {
    q["id"]: q for q in (HISTORY_QUESTIONS + QUESTIONNAIRE_QUESTIONS)
}


def _question_lines_compact(questions: list[dict]) -> str:
    lines: list[str] = []
    for q in questions:
        labels = " | ".join(str(opt["label"]) for opt in q["options"])
        lines.append(
            f"{q['id']} | {q['type']} | {q['question']} | 選項: {labels}"
        )
    return "\\n".join(lines)


def _questionnaire_prompt(source_name: str) -> str:
    history_lines = _question_lines_compact(HISTORY_QUESTIONS)
    questionnaire_lines = _question_lines_compact(QUESTIONNAIRE_QUESTIONS)

    return f"""
你是 CoachOS Scanner API 的「CoachOS v2.1 標準問卷」勾選辨識器。
資料來源：{source_name}

你只需要辨識每一題的「題號」與「實際被勾選的選項文字」。
不要直接產生 Firebase 欄位，不要產生健康判斷、運動建議、風險分級或醫療診斷。

輸出規則：
1. answers 只放本次圖片中實際出現的 H01-H10 / Q01-Q22。
2. question_id 必須使用標準題號，例如 H04、Q11。
3. selected_labels 必須使用下方列出的「原始選項文字」，不要改寫、翻譯或換成 canonical key；尤其數字範圍請原樣保留「–」，斜線請原樣保留「／」。
4. visibility="complete"：題幹與該題全部選項皆清楚完整可見，可可靠判斷哪些有勾、哪些未勾。
5. visibility="partial"：題目有出現，但有部分選項被裁切、遮擋或看不清楚；只輸出可確定被勾選的 selected_labels，並把不可靠選項放 unreadable_labels。
6. visibility="unreadable"：整題無法可靠判讀；selected_labels 留空，並在 unreadable_or_ambiguous_items 記錄題號。
7. 空白 checkbox 不可當成勾選；只有清楚打勾、填黑、圈選或明確標記才放入 selected_labels。
8. H04 / H05 / H09 的「無」就是一般可辨識選項；若同時勾「無」與其他項目，兩者都照實放入 selected_labels，Python 後端會處理「其他項目優先、無失效」。
9. H10 只需照實辨識最近手術時間；Python 後端會處理 no_surgery_history 規則。
10. 未出現在本次照片的題目不要放進 answers，避免補拍時把舊資料誤改成 false。
11. basic_info 只有清楚可見才擷取，否則填 null。

[A. 健康與運動相關資訊]
{history_lines}

[B. 運動與生活偏好]
{questionnaire_lines}
"""


_OPTION_PUNCT_TRANSLATION = str.maketrans({
    # Dash / range separators: OCR/LLM frequently alternates these glyphs.
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "﹣": "-",
    "－": "-",
    "~": "-",
    "～": "-",
    "〜": "-",
    # Slash variants. Canonical internal representation uses Chinese fullwidth slash.
    "/": "／",
    "∕": "／",
    "⁄": "／",
    # Comma variants used by a small number of questionnaire labels.
    ",": "、",
    "，": "、",
})

_OPTION_WRAPPER_CHARS = (
    "□☐☑☒✓✔√●○◯"
    "\"'“”‘’「」『』"
)


def _normalize_option_label(text: str) -> str:
    """Normalize typographic/OCR variants only; never perform semantic guessing."""
    import unicodedata

    value = unicodedata.normalize("NFKC", str(text or "")).strip()

    # Remove invisible formatting characters that can be introduced by OCR/copy-paste.
    value = (
        value.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
    )

    # NFKC can turn fullwidth punctuation into ASCII, therefore translate after NFKC.
    value = value.translate(_OPTION_PUNCT_TRANSLATION)

    # Ignore whitespace differences such as "30 - 60 分鐘" vs "30–60 分鐘".
    value = "".join(value.split())

    # Gemini occasionally echoes checkbox/quote wrappers even though the schema asks
    # for label text only. Removing wrappers is deterministic and does not change meaning.
    value = value.strip(_OPTION_WRAPPER_CHARS)

    # Only one standard option contains Latin text ("App"); casing is not semantic here.
    value = value.casefold()
    return value


def _normalize_question_id(value: str) -> str:
    """Normalize harmless OCR variants such as Ｑ０１ / Q1 / 'Q01:'."""
    import re
    import unicodedata

    raw = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    raw = "".join(raw.split())
    raw = raw.strip(":：.-_")
    match = re.fullmatch(r"([HQ])0*(\d{1,2})", raw)
    if not match:
        return raw
    number = int(match.group(2))
    return f"{match.group(1)}{number:02d}"


def _option_norms_for_question(qid: str) -> set[str]:
    question = ALL_STANDARD_QUESTIONS.get(qid)
    if not isinstance(question, dict):
        return set()
    return {
        _normalize_option_label(opt.get("label"))
        for opt in (question.get("options") or [])
        if isinstance(opt, dict)
    }


def _resolved_mapping_warning(item: object) -> bool:
    """True when an old '無法對應' warning is now resolvable by typographic normalization."""
    import re

    raw_item = str(item or "").strip()
    match = re.fullmatch(
        r"([HQ]\d{1,2})\s*無法對應的(?:勾選文字|模糊選項)：(.+)",
        raw_item,
    )
    if not match:
        return False

    qid = _normalize_question_id(match.group(1))
    raw_label = match.group(2).strip()
    return _normalize_option_label(raw_label) in _option_norms_for_question(qid)


def _drop_resolved_mapping_warnings(items: list | None) -> list:
    """Remove only obsolete exact-mapping warnings; keep genuinely ambiguous warnings."""
    output = []
    seen = set()
    for item in items or []:
        text_item = str(item)
        if _resolved_mapping_warning(text_item):
            continue
        if text_item not in seen:
            seen.add(text_item)
            output.append(text_item)
    return output


def _blank_canonical_questionnaire_result() -> dict:
    return {
        "detected_document_type": "other",
        "basic_info": {
            "name_on_form": None,
            "age": None,
            "gender": None,
            "student_number_on_form": None,
        },
        "health_profile": {key: None for key in HISTORY_FIELD_LABELS},
        "lifestyle_profile": {key: None for key in QUESTIONNAIRE_FIELD_LABELS},
        "control_selections": {
            "h04_none_selected": None,
            "h05_none_selected": None,
            "h09_none_selected": None,
        },
        "unreadable_or_ambiguous_items": [],
    }


def _compact_to_canonical(compact: dict) -> dict:
    """把小型 Gemini 輸出 deterministic 地轉回既有 HIS/QST boolean schema。"""
    result = _blank_canonical_questionnaire_result()
    result["basic_info"].update(compact.get("basic_info") or {})
    warnings = list(compact.get("unreadable_or_ambiguous_items") or [])

    seen_h = False
    seen_q = False

    for answer in compact.get("answers") or []:
        qid = _normalize_question_id(answer.get("question_id"))
        question = ALL_STANDARD_QUESTIONS.get(qid)
        if question is None:
            warnings.append(f"未知題號：{qid or '(空白)'}")
            continue

        if qid.startswith("H"):
            seen_h = True
            target = result["health_profile"]
        else:
            seen_q = True
            target = result["lifestyle_profile"]

        visibility = str(answer.get("visibility") or "complete").lower()
        selected_raw = [str(x) for x in (answer.get("selected_labels") or [])]
        unreadable_raw = [str(x) for x in (answer.get("unreadable_labels") or [])]

        option_by_norm = {
            _normalize_option_label(opt["label"]): opt
            for opt in question["options"]
        }
        selected_norm = {_normalize_option_label(x) for x in selected_raw}
        unreadable_norm = {_normalize_option_label(x) for x in unreadable_raw}

        # Gemini 若回傳非標準選項文字，不猜測 mapping，僅標記人工確認。
        for raw, norm in zip(selected_raw, [_normalize_option_label(x) for x in selected_raw]):
            if norm not in option_by_norm:
                warnings.append(f"{qid} 無法對應的勾選文字：{raw}")
        for raw, norm in zip(unreadable_raw, [_normalize_option_label(x) for x in unreadable_raw]):
            if norm not in option_by_norm:
                warnings.append(f"{qid} 無法對應的模糊選項：{raw}")

        valid_selected = selected_norm & set(option_by_norm)
        valid_unreadable = unreadable_norm & set(option_by_norm)

        # 完整可見時才允許把「清楚未勾選」轉成 false。
        # 單選題若完全沒有任何勾選，視為未作答，保留 null，不自行推成 false。
        if visibility == "complete":
            if question["type"] == "multiple":
                for opt in question["options"]:
                    if "control_key" in opt:
                        result["control_selections"][opt["control_key"]] = False
                    elif opt.get("key"):
                        target[opt["key"]] = False
            elif valid_selected:
                # 一般單選題：選中項 true，其餘清楚可見項 false。
                # Yes/No 題共用同一 canonical key，稍後由被選中的 value 決定 true/false。
                keys = [opt.get("key") for opt in question["options"] if opt.get("key")]
                if len(set(keys)) > 1:
                    for opt in question["options"]:
                        if opt.get("key"):
                            target[opt["key"]] = False

        # 模糊項目優先保留 null，避免 complete 預填 false 覆蓋不可靠影像。
        for norm in valid_unreadable:
            opt = option_by_norm[norm]
            if "control_key" in opt:
                result["control_selections"][opt["control_key"]] = None
            elif opt.get("key"):
                target[opt["key"]] = None
            warnings.append(f"{qid} 選項辨識不清：{opt['label']}")

        # 實際勾選永遠優先於未勾選/模糊狀態。
        for norm in valid_selected:
            opt = option_by_norm[norm]
            if "control_key" in opt:
                result["control_selections"][opt["control_key"]] = True
            elif opt.get("key"):
                target[opt["key"]] = bool(opt.get("value", True))

        if visibility == "unreadable":
            warnings.append(f"{qid} 整題無法可靠辨識")

    if seen_h and seen_q:
        result["detected_document_type"] = "mixed"
    elif seen_h:
        result["detected_document_type"] = "history"
    elif seen_q:
        result["detected_document_type"] = "questionnaire"

    # 去重並保留順序；同時移除已可由純字形正規化可靠對應的舊式假警告。
    result["unreadable_or_ambiguous_items"] = _drop_resolved_mapping_warnings(warnings)
    return result

H04_CONDITION_KEYS = [
    "hypertension", "hypotension", "heart_related_condition",
    "hypoglycemia_symptom", "kidney_related_condition",
    "liver_related_condition", "lung_related_condition", "osteoporosis",
    "parkinsons_disease", "neurological_condition",
]

H05_DISCOMFORT_KEYS = [
    "discomfort_neck", "discomfort_shoulder", "discomfort_elbow",
    "discomfort_wrist_hand", "discomfort_upper_back", "discomfort_low_back",
    "discomfort_hip", "discomfort_knee", "discomfort_ankle_foot",
]

H09_SURGERY_LOCATION_KEYS = [
    "surgery_cervical_spine", "surgery_shoulder", "surgery_elbow",
    "surgery_wrist_hand", "surgery_thoracic_spine", "surgery_lumbar_spine",
    "surgery_hip", "surgery_knee", "surgery_ankle_foot",
]

H10_SURGERY_RECENCY_KEYS = [
    "surgery_most_recent_within_3_months",
    "surgery_most_recent_3_6_months",
    "surgery_most_recent_6_12_months",
    "surgery_most_recent_over_1_year",
]


def _apply_history_none_conflict_rules(extraction: dict) -> dict:
    """
    套用紙本「無」選項的 deterministic conflict rules。

    規則：
    - H04/H05：若「無」與任何具體選項同時勾選，具體選項優先，「無」失效。
      若只勾「無」，該題所有 canonical keys=false。
    - H09：若「無」與任何手術部位同時勾選，部位優先，「無」失效。
    - 只要出現任何手術部位=true 或 H10 具體最近手術時間=true，
      no_surgery_history 一律強制為 false。
    - 只有 H09 清楚勾「無」，且沒有任何手術部位/最近手術時間證據時，
      no_surgery_history 才設為 true。
    """
    health = dict(extraction.get("health_profile") or {})
    controls = dict(extraction.get("control_selections") or {})

    # H04: 「無」只在沒有任何具體健康狀況被勾選時生效。
    if controls.get("h04_none_selected") is True:
        if any(health.get(k) is True for k in H04_CONDITION_KEYS):
            controls["h04_none_selected"] = False
        else:
            for key in H04_CONDITION_KEYS:
                health[key] = False

    # H05: 「無」只在沒有任何具體不適部位被勾選時生效。
    if controls.get("h05_none_selected") is True:
        if any(health.get(k) is True for k in H05_DISCOMFORT_KEYS):
            controls["h05_none_selected"] = False
        else:
            for key in H05_DISCOMFORT_KEYS:
                health[key] = False

    surgery_location_selected = any(
        health.get(k) is True for k in H09_SURGERY_LOCATION_KEYS
    )
    surgery_recency_selected = any(
        health.get(k) is True for k in H10_SURGERY_RECENCY_KEYS
    )

    # 有任何手術的正向證據時，「沒有手術史」必須失效。
    if surgery_location_selected or surgery_recency_selected:
        health["no_surgery_history"] = False
        if controls.get("h09_none_selected") is True:
            controls["h09_none_selected"] = False
    elif controls.get("h09_none_selected") is True:
        # H09 單獨勾「無」：所有手術部位=false，沒有手術史=true。
        for key in H09_SURGERY_LOCATION_KEYS:
            health[key] = False
        health["no_surgery_history"] = True

    extraction["health_profile"] = health
    extraction["control_selections"] = controls
    return extraction


def _parse_questionnaire_compact_response(response) -> dict:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, QuestionnaireCompactExtraction):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return QuestionnaireCompactExtraction.model_validate(parsed).model_dump()
    raw_text = (response.text or "").strip()
    return QuestionnaireCompactExtraction.model_validate_json(raw_text).model_dump()


def _prepare_questionnaire_image(image: Image.Image) -> bytes:
    """縮小手機原圖，保留 checkbox/小字辨識所需解析度，同時避免 inline payload 過大。"""
    img = image.convert("RGB")
    max_side = 2200
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def extract_questionnaire_from_images(
    images: list[Image.Image],
    metric_context: dict[str, Any] | None = None,
) -> dict:
    if not images:
        raise HTTPException(status_code=422, detail="至少需要一張 CoachOS 標準問卷照片")
    if len(images) > 4:
        raise HTTPException(status_code=422, detail="單次最多上傳 4 張問卷照片")

    client = app.state.genai_client
    contents = []
    total_bytes = 0

    for image in images:
        data = _prepare_questionnaire_image(image)
        total_bytes += len(data)
        contents.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

    contents.append(_questionnaire_prompt("CoachOS 標準紙本問卷 / 手機拍照"))

    logger.info(
        f"📝 CoachOS 問卷送 Gemini：pages={len(images)}, "
        f"inline_images={total_bytes / 1024 / 1024:.2f} MB, model={GEMINI_MODEL}"
    )

    response = _generate_content_with_retry(
        client,
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            # 重要：只要求 Gemini 回傳小型「題號 + 勾選文字」schema。
            # 39 + 153 個 Firebase boolean 改由 Python deterministic mapping。
            response_schema=QuestionnaireCompactExtraction,
        ),
        operation="questionnaire_recognition",
        metric_context={
            **(metric_context or {}),
            "prompt_version": f"questionnaire_compact_{QUESTIONNAIRE_VERSION}_20260929",
            "schema_output_version": QUESTIONNAIRE_VERSION,
            "image_count": len(images),
            "image_bytes": total_bytes,
        },
    )

    try:
        compact = _parse_questionnaire_compact_response(response)
        result = _compact_to_canonical(compact)
        result = _apply_history_none_conflict_rules(result)
        logger.info(
            f"✅ CoachOS 標準問卷辨識完成，pages={len(images)}, "
            f"answers={len(compact.get('answers') or [])}, model={GEMINI_MODEL}"
        )
        return result
    except Exception as e:
        raw_text = ""
        try:
            raw_text = (response.text or "").strip()
        except Exception:
            pass
        logger.error(
            f"CoachOS 標準問卷 Structured Output 內容解析失敗: {e}; "
            f"pages={len(images)}, inline_mb={total_bytes / 1024 / 1024:.2f}, "
            f"raw_response={raw_text[:2000]!r}"
        )
        raise HTTPException(
            status_code=500,
            detail="AI 已回應，但 CoachOS 標準問卷結構化資料解析失敗",
        ) from e


# -----------------------------------------------------------------------------
# 4. 資料品質驗證
# -----------------------------------------------------------------------------
# 這些範圍只用於攔截 OCR/AI 明顯錯讀，不是醫療診斷標準。
NUMERIC_ALLOWED_RANGES = {
    "height_m": (1.00, 2.50),
    "weight_kg": (20.0, 350.0),
    "body_fat_percentage": (1.0, 75.0),
    "body_fat_mass_kg": (0.2, 200.0),
    "fat_free_mass_kg": (5.0, 250.0),
    "basal_metabolic_rate": (400.0, 5000.0),
    "smi_kg_m2": (1.0, 20.0),
    "visceral_fat_value": (0.1, 100.0),
    "visceral_fat_reference_low": (0.0, 100.0),
    "visceral_fat_reference_high": (0.1, 100.0),
    "inbody_score": (0.0, 200.0),
    "segmental_muscle_ra": (0.1, 15.0),
    "segmental_muscle_la": (0.1, 15.0),
    "segmental_muscle_rl": (0.2, 30.0),
    "segmental_muscle_ll": (0.2, 30.0),
}



def _to_finite_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_gender_optional(gender) -> str | None:
    """將 m/f、male/female、男/女統一成 male/female；供 BCS 與內部判斷使用。"""
    value = str(gender or "").strip().lower()
    if value in {"male", "m", "男", "男性"}:
        return "male"
    if value in {"female", "f", "女", "女性"}:
        return "female"
    return None


def gender_to_firestore_code(gender) -> str | None:
    """Body composition Firestore 儲存格式：male -> m、female -> f；未知維持 None。"""
    normalized = normalize_gender_optional(gender)
    if normalized == "male":
        return "m"
    if normalized == "female":
        return "f"
    return None


def validate_health_data(raw_result: dict):
    result = dict(raw_result)
    warnings = []

    device_type = str(result.get("device_type", "other")).strip().lower()
    if device_type not in {"inbody", "tanita", "other"}:
        warnings.append(f"device_type='{result.get('device_type')}' 不合法，已改為 other")
        device_type = "other"
    result["device_type"] = device_type

    # gender 必須直接來自報表 OCR；只做格式標準化，不從其他資料源補值。
    raw_gender = result.get("gender")
    normalized_gender = normalize_gender_optional(raw_gender)
    if raw_gender is not None and normalized_gender is None:
        warnings.append(f"gender='{raw_gender}' 無法辨識為 male/female，已設為 null")
    result["gender"] = normalized_gender

    # age 只允許報表 OCR 直接讀取；目前 CoachOS Rule Base 支援 18+。
    raw_age = result.get("age")
    if raw_age is None:
        result["age"] = None
    else:
        try:
            age_value = int(float(raw_age))
            if 18 <= age_value <= 120:
                result["age"] = age_value
            else:
                warnings.append(f"age={raw_age} 超出 OCR 防呆範圍 [18, 120]，已設為 null")
                result["age"] = None
        except (TypeError, ValueError):
            warnings.append(f"age='{raw_age}' 不是有效年齡，已設為 null")
            result["age"] = None

    # 僅允許 schema 中的數值欄位；未出現欄位統一補 null。
    for field, (minimum, maximum) in NUMERIC_ALLOWED_RANGES.items():
        original_value = result.get(field)
        if original_value is None:
            result[field] = None
            continue

        value = _to_finite_float(original_value)
        if value is None:
            warnings.append(f"{field}='{original_value}' 不是有效數字，已設為 null")
            result[field] = None
        elif not minimum <= value <= maximum:
            warnings.append(
                f"{field}={value} 超出 OCR 防呆範圍 [{minimum}, {maximum}]，已設為 null"
            )
            result[field] = None
        else:
            result[field] = value

    vf_low = result.get("visceral_fat_reference_low")
    vf_high = result.get("visceral_fat_reference_high")
    if vf_low is not None and vf_high is not None and vf_low >= vf_high:
        warnings.append("visceral_fat_reference_low >= high，兩者已設為 null")
        result["visceral_fat_reference_low"] = None
        result["visceral_fat_reference_high"] = None

    # InBody Score 僅 InBody 保存。
    if device_type != "inbody":
        result["inbody_score"] = None

    # InBody 的 Visceral Fat Level 圖只有 level 與視覺區間標示，
    # 不把 Low/High 或 <10 解讀成可用的 reference range。
    # InBody 僅保存 visceral_fat_value。
    if device_type == "inbody":
        result["visceral_fat_reference_low"] = None
        result["visceral_fat_reference_high"] = None

    return result, warnings


# -----------------------------------------------------------------------------
# 5. 共通衍生指標
# -----------------------------------------------------------------------------
def calculate_derived_metrics(result: dict) -> dict:
    height_m = result.get("height_m")
    weight_kg = result.get("weight_kg")

    ra = result.get("segmental_muscle_ra")
    la = result.get("segmental_muscle_la")
    rl = result.get("segmental_muscle_rl")
    ll = result.get("segmental_muscle_ll")

    bmi = None
    if height_m is not None and height_m > 0 and weight_kg is not None:
        bmi = round(weight_kg / (height_m ** 2), 2)

    # ALM 與由 ALM 推算的 SMI：四肢資料完整時才計算。
    alm = None
    calculated_smi = None
    if all(v is not None for v in [ra, la, rl, ll]):
        alm = round(ra + la + rl + ll, 2)
        if height_m is not None and height_m > 0:
            calculated_smi = round(alm / (height_m ** 2), 2)

    # 上下肢 AI 獨立計算：避免其中一組缺值讓另一組也失效。
    ai_upper = None
    if ra is not None and la is not None:
        upper_avg = (ra + la) / 2
        ai_upper = round((abs(ra - la) / upper_avg) * 100, 2) if upper_avg > 0 else 0.0

    ai_lower = None
    if rl is not None and ll is not None:
        lower_avg = (rl + ll) / 2
        ai_lower = round((abs(rl - ll) / lower_avg) * 100, 2) if lower_avg > 0 else 0.0

    return {
        "BMI": bmi,
        "ALM": alm,
        "SMI": calculated_smi,
        "AI_upper_pct": ai_upper,
        "AI_lower_pct": ai_lower,
    }


# -----------------------------------------------------------------------------
# 6. CoachOS Body Composition Score (BCS v3.0 rule-calibrated)
# -----------------------------------------------------------------------------
#
# 設計原則：
# 1) 以 InBody 公開說明作為方向，而不是宣稱複製其專有公式：
#    - 基準分 80。
#    - 標準體重依身高與性別的 desirable BMI：male 22、female 21.5。
#    - 理想體脂比例：male 15%、female 23%。
#    - 比較目前 Lean Body Mass / Body Fat Mass 與標準值的差異。
# 2) 再疊加 CoachOS 現有 rule_key 中「體組成直接相關」的四項扣分：
#    Body Fat / SMI / AI_lower / AI_upper。
# 3) Knee / Ankle 屬動作功能，不納入 Body Composition Score。
# 4) 下列 calibration factor 是以 2026-09-30 test-run 樣本做產品校準，
#    並非 InBody 官方公開的專有係數；後續應隨真實樣本增加持續驗證。
#
BCS_VERSION = "BCS_v3.0_rule_calibrated_20260930"
BCS_BASELINE = 80.0

# InBody 公開說明使用的標準體重 / 理想體脂基準。
BCS_STANDARD_BMI = {"male": 22.0, "female": 21.5}
BCS_IDEAL_BODY_FAT_RATIO = {"male": 0.15, "female": 0.23}

# CoachOS rule_key 現行體組成門檻。
SMI_REFERENCE = {"male": 7.0, "female": 5.7}
RULE_BODY_FAT_HIGH_PCT = {"male": 25.0, "female": 30.0}
RULE_AI_WARN_PCT = 10.0

# BCS v3 calibration factors（points per kg）。
# test-run 校準目標：降低 v2.3 系統性偏高，並貼近目前 InBody Score 分布。
BCS_CALIBRATION = {
    "lean_mass_kg_factor": 1.10,
    "excess_fat_kg_factor": 0.85,
    "fat_deficit_kg_factor": 0.25,
    "fat_deficit_reward_cap": 3.0,
}

# Rule Key penalty：固定、透明、可審核。
# Body Fat / SMI 已在連續值公式中反映，因此 penalty 採小幅額外扣分；
# AI_lower / AI_upper 是 CoachOS 額外納入的左右肌肉平衡訊號。
BCS_RULE_PENALTIES = {
    "body_fat_high": 1.0,
    "smi_low": 3.0,
    "ai_lower_warn": 3.0,
    "ai_upper_warn": 3.0,
}

# 為兼容既有前端，仍保留 body-fat reference 與 muscle/body_fat component 顯示。
PBF_DEFAULTS = {
    "male": {"low": 10.0, "ideal": 15.0, "high": 20.0},
    "female": {"low": 18.0, "ideal": 23.0, "high": 28.0},
}


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _finite_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def resolve_body_fat_reference(gender: str) -> dict:
    """保留既有前端 / Firestore reference schema；BCS v3 主公式不直接使用 low/high。"""
    default = PBF_DEFAULTS[gender]
    return {
        "low_pct": float(default["low"]),
        "ideal_pct": float(default["ideal"]),
        "high_pct": float(default["high"]),
        "source": "fixed_gender_reference",
    }


def _resolve_bcs_inputs(result: dict, derived_metrics: dict) -> dict:
    """整理 BCS v3 所需的可追溯輸入；不修改原始 OCR measurement。"""
    height_m = _finite_number(result.get("height_m"))
    weight_kg = _finite_number(result.get("weight_kg"))
    pbf = _finite_number(result.get("body_fat_percentage"))
    bfm = _finite_number(result.get("body_fat_mass_kg"))
    ffm = _finite_number(result.get("fat_free_mass_kg"))

    # Body Fat Mass fallback：僅供 BCS 計算，不回寫 measurement。
    bfm_source = "reported_body_fat_mass"
    if bfm is None and weight_kg is not None and pbf is not None:
        bfm = weight_kg * pbf / 100.0
        bfm_source = "derived_weight_x_pbf"
    elif bfm is None:
        bfm_source = "unavailable"

    # Lean Body Mass 優先用 FFM；沒有時使用 Weight - Body Fat Mass。
    lbm = ffm
    lbm_source = "reported_fat_free_mass"
    if lbm is None and weight_kg is not None and bfm is not None:
        lbm = weight_kg - bfm
        lbm_source = "derived_weight_minus_bfm"
    elif lbm is None:
        lbm_source = "unavailable"

    reported_smi = _finite_number(result.get("smi_kg_m2"))
    calculated_smi = _finite_number(derived_metrics.get("SMI"))
    if reported_smi is not None:
        scoring_smi = reported_smi
        smi_source = "reported_smi"
    elif calculated_smi is not None:
        scoring_smi = calculated_smi
        smi_source = "segmental_alm_over_height2"
    else:
        scoring_smi = None
        smi_source = "unavailable"

    return {
        "height_m": height_m,
        "weight_kg": weight_kg,
        "body_fat_percentage": pbf,
        "body_fat_mass_kg": bfm,
        "body_fat_mass_source": bfm_source,
        "lean_body_mass_kg": lbm,
        "lean_body_mass_source": lbm_source,
        "smi_kg_m2": scoring_smi,
        "smi_source": smi_source,
        "ai_lower_pct": _finite_number(derived_metrics.get("AI_lower_pct")),
        "ai_upper_pct": _finite_number(derived_metrics.get("AI_upper_pct")),
    }


def _bodycomp_rule_statuses(gender: str, inputs: dict) -> dict:
    """依目前 CoachOS rule_key 規格，產生 BCS 相關四個 rule status。"""
    pbf = inputs.get("body_fat_percentage")
    smi = inputs.get("smi_kg_m2")
    ai_lower = inputs.get("ai_lower_pct")
    ai_upper = inputs.get("ai_upper_pct")

    bodyfat_status = None
    if pbf is not None:
        bodyfat_status = (
            "normal"
            if pbf <= RULE_BODY_FAT_HIGH_PCT[gender]
            else "high"
        )

    smi_status = None
    if smi is not None:
        smi_status = "normal" if smi >= SMI_REFERENCE[gender] else "low"

    ai_lower_status = None
    if ai_lower is not None:
        ai_lower_status = "normal" if ai_lower < RULE_AI_WARN_PCT else "warn"

    ai_upper_status = None
    if ai_upper is not None:
        ai_upper_status = "normal" if ai_upper < RULE_AI_WARN_PCT else "warn"

    return {
        "body_fat": bodyfat_status,
        "smi": smi_status,
        "ai_lower": ai_lower_status,
        "ai_upper": ai_upper_status,
    }


def _rule_penalty_breakdown(statuses: dict) -> dict:
    applied = {
        "body_fat_high": (
            BCS_RULE_PENALTIES["body_fat_high"]
            if statuses.get("body_fat") == "high"
            else 0.0
        ),
        "smi_low": (
            BCS_RULE_PENALTIES["smi_low"]
            if statuses.get("smi") == "low"
            else 0.0
        ),
        "ai_lower_warn": (
            BCS_RULE_PENALTIES["ai_lower_warn"]
            if statuses.get("ai_lower") == "warn"
            else 0.0
        ),
        "ai_upper_warn": (
            BCS_RULE_PENALTIES["ai_upper_warn"]
            if statuses.get("ai_upper") == "warn"
            else 0.0
        ),
    }
    return {
        "applied": applied,
        "total": round(sum(applied.values()), 2),
    }


def calculate_coachos_bcs(
    result: dict,
    derived_metrics: dict,
    gender: str | None,
    warnings: list[str],
) -> dict:
    """
    CoachOS BCS v3.0：80 分基準 + LBM/BFM 校準 + rule_key penalty。

    公式（產品校準版，不宣稱等同 InBody 專有公式）：
      standard_weight = height² * desirable_BMI
      standard_BFM = standard_weight * ideal_fat_ratio
      standard_LBM = standard_weight - standard_BFM

      lean_adjustment = (current_LBM - standard_LBM) * 1.10
      fat_adjustment =
          - max(current_BFM - standard_BFM, 0) * 0.85
          + min(max(standard_BFM - current_BFM, 0) * 0.25, 3.0)

      BCS = 80 + lean_adjustment + fat_adjustment - rule_penalty

    Rule penalty：
      body_fat high  -1
      SMI low        -3
      AI_lower warn  -3
      AI_upper warn  -3

    注意：Knee / Ankle 不屬於體組成，不納入本分數。
    """
    inputs = _resolve_bcs_inputs(result, derived_metrics)

    if gender not in BCS_STANDARD_BMI:
        warnings.append(
            "報表 gender 缺失或無法辨識；無法選定 BCS v3 性別參考值，因此 CoachOS 體組成分數不計算"
        )
        return {
            "name": "CoachOS Body Composition Score",
            "value": None,
            "version": BCS_VERSION,
            "status": "insufficient_gender",
            "coverage": 0.0,
            "components": {"muscle": None, "body_fat": None},
            "weights": dict(BCS_CALIBRATION),
            "rule": {
                "signature": None,
                "statuses": {},
                "penalties": dict(BCS_RULE_PENALTIES),
                "applied_penalties": {},
                "total_penalty": None,
            },
            "references": {"gender": gender},
        }

    height_m = inputs.get("height_m")
    current_lbm = inputs.get("lean_body_mass_kg")
    current_bfm = inputs.get("body_fat_mass_kg")

    if height_m is None or height_m <= 0 or current_lbm is None or current_bfm is None:
        warnings.append(
            "BCS v3 缺少 height / lean body mass / body fat mass 必要資料，因此 CoachOS 體組成分數不計算"
        )
        return {
            "name": "CoachOS Body Composition Score",
            "value": None,
            "version": BCS_VERSION,
            "status": "insufficient_data",
            "coverage": 0.0,
            "components": {"muscle": None, "body_fat": None},
            "weights": dict(BCS_CALIBRATION),
            "rule": {
                "signature": None,
                "statuses": {},
                "penalties": dict(BCS_RULE_PENALTIES),
                "applied_penalties": {},
                "total_penalty": None,
            },
            "references": {
                "gender": gender,
                "height_m": height_m,
                "lean_body_mass_kg": current_lbm,
                "body_fat_mass_kg": current_bfm,
            },
        }

    standard_bmi = BCS_STANDARD_BMI[gender]
    ideal_fat_ratio = BCS_IDEAL_BODY_FAT_RATIO[gender]
    standard_weight = (height_m ** 2) * standard_bmi
    standard_bfm = standard_weight * ideal_fat_ratio
    standard_lbm = standard_weight - standard_bfm

    lean_diff_kg = current_lbm - standard_lbm
    fat_diff_kg = current_bfm - standard_bfm
    excess_fat_kg = max(fat_diff_kg, 0.0)
    fat_deficit_kg = max(-fat_diff_kg, 0.0)

    lean_adjustment = lean_diff_kg * BCS_CALIBRATION["lean_mass_kg_factor"]
    excess_fat_adjustment = -excess_fat_kg * BCS_CALIBRATION["excess_fat_kg_factor"]
    fat_deficit_reward = min(
        fat_deficit_kg * BCS_CALIBRATION["fat_deficit_kg_factor"],
        BCS_CALIBRATION["fat_deficit_reward_cap"],
    )
    fat_adjustment = excess_fat_adjustment + fat_deficit_reward

    statuses = _bodycomp_rule_statuses(gender, inputs)
    penalty = _rule_penalty_breakdown(statuses)

    raw_score = (
        BCS_BASELINE
        + lean_adjustment
        + fat_adjustment
        - penalty["total"]
    )
    score = round(clamp(raw_score), 1)

    # 相容既有 Scanner Web：仍提供 muscle / body_fat 兩個 component 顯示值。
    muscle_component = round(clamp(BCS_BASELINE + lean_adjustment), 2)
    body_fat_component = round(clamp(BCS_BASELINE + fat_adjustment), 2)

    available_rule_fields = sum(v is not None for v in statuses.values())
    coverage = 0.60 + available_rule_fields * 0.10
    status = "complete" if available_rule_fields == 4 else "partial"

    signature = "_".join(
        statuses.get(k) or "missing"
        for k in ("body_fat", "smi", "ai_lower", "ai_upper")
    )

    body_fat_reference = resolve_body_fat_reference(gender)

    return {
        "name": "CoachOS Body Composition Score",
        "value": score,
        "version": BCS_VERSION,
        "status": status,
        "coverage": round(coverage, 4),
        "components": {
            "muscle": muscle_component,
            "body_fat": body_fat_component,
            "baseline": BCS_BASELINE,
            "lean_mass_adjustment": round(lean_adjustment, 3),
            "fat_mass_adjustment": round(fat_adjustment, 3),
            "rule_penalty": penalty["total"],
        },
        "weights": {
            "lean_mass_kg_factor": BCS_CALIBRATION["lean_mass_kg_factor"],
            "excess_fat_kg_factor": BCS_CALIBRATION["excess_fat_kg_factor"],
            "fat_deficit_kg_factor": BCS_CALIBRATION["fat_deficit_kg_factor"],
            "fat_deficit_reward_cap": BCS_CALIBRATION["fat_deficit_reward_cap"],
        },
        "component_sources": {
            "lean_body_mass": inputs.get("lean_body_mass_source"),
            "body_fat_mass": inputs.get("body_fat_mass_source"),
            "smi": inputs.get("smi_source"),
            "ai_lower": "derived_segmental_muscle" if inputs.get("ai_lower_pct") is not None else "unavailable",
            "ai_upper": "derived_segmental_muscle" if inputs.get("ai_upper_pct") is not None else "unavailable",
        },
        "rule": {
            "signature": signature,
            "statuses": statuses,
            "thresholds": {
                "body_fat_high_pct": RULE_BODY_FAT_HIGH_PCT[gender],
                "smi_low_below_kg_m2": SMI_REFERENCE[gender],
                "ai_warn_at_or_above_pct": RULE_AI_WARN_PCT,
            },
            "penalties": dict(BCS_RULE_PENALTIES),
            "applied_penalties": penalty["applied"],
            "total_penalty": penalty["total"],
        },
        "references": {
            "gender": gender,
            "standard_bmi": standard_bmi,
            "ideal_body_fat_ratio": ideal_fat_ratio,
            "standard_weight_kg": round(standard_weight, 3),
            "standard_lbm_kg": round(standard_lbm, 3),
            "standard_bfm_kg": round(standard_bfm, 3),
            "current_lbm_kg": round(current_lbm, 3),
            "current_bfm_kg": round(current_bfm, 3),
            "lean_mass_diff_kg": round(lean_diff_kg, 3),
            "body_fat_mass_diff_kg": round(fat_diff_kg, 3),
            "smi_reference_kg_m2": SMI_REFERENCE.get(gender),
            "smi_used_for_bcs": inputs.get("smi_kg_m2"),
            "body_fat": body_fat_reference,
        },
        "calibration": {
            "method": "inbody_public_logic_plus_coachos_rule_penalty",
            "baseline": BCS_BASELINE,
            "test_run_date": "2026-09-30",
            "note": "Product calibration only; not the proprietary InBody scoring formula.",
        },
    }


# -----------------------------------------------------------------------------
# 7. 建立 Firestore payload（獨立函數，方便測試/模擬）
# -----------------------------------------------------------------------------
def build_firebase_data(
    result: dict,
    company_id: str,
    branch_id: str,
    coach_id: str,
    target_user_id: str,
    user_name: str,
    validation_warnings: list[str],
) -> dict:
    derived_metrics = calculate_derived_metrics(result)
    # 傳入同一個 warnings list，BCS 資料不足等狀態可記錄在 data_quality。
    report_gender = result.get("gender")
    coachos_score = calculate_coachos_bcs(
        result, derived_metrics, report_gender, validation_warnings
    )

    completeness = {
        "age_available": result.get("age") is not None,
        "gender_available": result.get("gender") is not None,
        "height_available": result.get("height_m") is not None,
        "weight_available": result.get("weight_kg") is not None,
        "body_fat_available": result.get("body_fat_percentage") is not None,
        "reported_smi_available": result.get("smi_kg_m2") is not None,
        "segmental_upper_complete": all(result.get(k) is not None for k in ["segmental_muscle_ra", "segmental_muscle_la"]),
        "segmental_lower_complete": all(result.get(k) is not None for k in ["segmental_muscle_rl", "segmental_muscle_ll"]),
        "segmental_all_complete": all(result.get(k) is not None for k in ["segmental_muscle_ra", "segmental_muscle_la", "segmental_muscle_rl", "segmental_muscle_ll"]),
        "alm_available": derived_metrics.get("ALM") is not None,
        "calculated_smi_available": derived_metrics.get("SMI") is not None,
        "visceral_fat_available": result.get("visceral_fat_value") is not None,
        "coachos_score_available": coachos_score.get("value") is not None,
    }

    local_now = datetime.now(TAIPEI_TZ)

    return {
        "company_id": company_id,
        "branch_id": branch_id,
        "coach_id": coach_id,
        "used_id": target_user_id,
        "user_name": user_name,
        # Body Composition age 直接由報表 OCR 取得，提供 Coach Rule age fallback。
        "age": result.get("age"),
        # Firestore body_composition gender 固定使用簡碼：m / f。
        # 內部 result 仍保留 male / female，避免影響 BCS 的性別參考表。
        "gender": gender_to_firestore_code(result.get("gender")),
        "gender_source": "report_ocr" if result.get("gender") else "unavailable",
        "type": "body_composition",
        "device_type": result.get("device_type"),
        "gemini_model": GEMINI_MODEL,
        "gemini_provider": "vertex_ai",

        "measurements": {
            "height_m": result.get("height_m"),
            "weight_kg": result.get("weight_kg"),
            "body_fat_percentage": result.get("body_fat_percentage"),
            "body_fat_mass_kg": result.get("body_fat_mass_kg"),
            "fat_free_mass_kg": result.get("fat_free_mass_kg"),
            "basal_metabolic_rate": result.get("basal_metabolic_rate"),
            "smi_kg_m2": result.get("smi_kg_m2"),
            "visceral_fat_value": result.get("visceral_fat_value"),
        },

        "segmental_muscle": {
            "ra": result.get("segmental_muscle_ra"),
            "la": result.get("segmental_muscle_la"),
            "rl": result.get("segmental_muscle_rl"),
            "ll": result.get("segmental_muscle_ll"),
        },

        "derived_metrics": derived_metrics,

        "scores": {
            # 保持固定 schema；Tanita / other 為 null，不存在 Tanita Vendor Score。
            "inbody_score": result.get("inbody_score") if result.get("device_type") == "inbody" else None,
            "coachos": coachos_score,
        },

        "data_quality": {
            "validation_passed": len(validation_warnings) == 0,
            "validation_warnings": validation_warnings,
            "completeness": completeness,
        },

        "record_date": local_now.strftime("%Y-%m-%d"),
        "recorded_at_local": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "createdAt": firestore.SERVER_TIMESTAMP,
    }


def sanitize_doc_id_part(value, fallback: str) -> str:
    """避免 / 等字元破壞 Firestore document path。"""
    text = str(value or fallback).strip()
    text = text.replace("/", "-").replace("\\", "-")
    return text or fallback


def require_context_id(value: str, field_name: str) -> str:
    """
    company/branch/coach/user 等識別值不可為空，也不可是 UNKNOWN_*。
    避免錯誤上下文被永久寫進 Firestore Document ID 與欄位。
    """
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.upper().startswith("UNKNOWN"):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} 缺失或無效，請由前端從 Students 傳入正確值",
        )
    return cleaned


def _clean_identity_text(value) -> str:
    return "".join(str(value or "").strip().lower().split())


def _is_empty_value(value) -> bool:
    """
    用於判斷 OCR / 表單欄位是否真的有內容。
    False / 0 是有效資訊，不視為空值。
    """
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "null", "none", "--"}

    if isinstance(value, list):
        return len(value) == 0 or all(_is_empty_value(v) for v in value)

    if isinstance(value, dict):
        return len(value) == 0 or all(_is_empty_value(v) for v in value.values())

    return False


def _has_meaningful_content(value) -> bool:
    return not _is_empty_value(value)


def _canonical_item_key(value) -> str:
    """list 去重使用；dict/list 轉成固定排序 JSON。"""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        return str(value)


def _merge_unique_lists(old_list, new_list) -> list:
    merged = []
    seen = set()

    for item in (old_list or []) + (new_list or []):
        key = _canonical_item_key(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)

    return merged


def _smart_merge_values(old_value, new_value, path: str, conflicts: list[dict]):
    """
    同一天補拍時的智慧合併：
    - dict：遞迴合併
    - list：保留舊資料 + 新資料，去重
    - 空的新值：不覆蓋舊值
    - 舊值為空：補入新值
    - scalar 衝突：採最新辨識值，同時留下 conflict 供人工追蹤
    """
    if _is_empty_value(new_value):
        return old_value

    if _is_empty_value(old_value):
        return new_value

    if isinstance(old_value, dict) and isinstance(new_value, dict):
        merged = dict(old_value)

        for key, new_child in new_value.items():
            child_path = f"{path}.{key}" if path else key
            merged[key] = _smart_merge_values(
                old_value.get(key),
                new_child,
                child_path,
                conflicts,
            )

        return merged

    if isinstance(old_value, list) and isinstance(new_value, list):
        return _merge_unique_lists(old_value, new_value)

    if old_value == new_value:
        return old_value

    conflicts.append(
        {
            "field": path,
            "previous_value": old_value,
            "new_value": new_value,
            "resolution": "latest_value_used",
        }
    )
    return new_value


def _validate_questionnaire_identity(
    extraction: dict,
    target_user_id: str,
    user_name: str,
) -> tuple[dict, list[str]]:
    """
    App 選取的 used_id / user_name 才是 Firebase 綁定真值。
    OCR 表單上的姓名、編號只拿來做一致性提醒。
    """
    basic = dict(extraction.get("basic_info") or {})
    warnings: list[str] = []

    age = basic.get("age")
    if age is not None and not (0 <= age <= 120):
        warnings.append(f"問卷 age={age} 超出合理資料範圍，已清除")
        basic["age"] = None

    form_name = basic.get("name_on_form")
    if form_name and user_name:
        if _clean_identity_text(form_name) != _clean_identity_text(user_name):
            warnings.append(
                f"問卷姓名「{form_name}」與目前選擇學員「{user_name}」不同，請人工確認"
            )

    form_student_id = basic.get("student_number_on_form")
    if form_student_id:
        if _clean_identity_text(form_student_id) != _clean_identity_text(target_user_id):
            warnings.append(
                f"問卷編號「{form_student_id}」與目前選擇 used_id「{target_user_id}」不同，請人工確認"
            )

    return basic, warnings


def _build_source_event(
    source: Literal["photo"],
    source_metadata: dict | None,
) -> dict:
    local_now = datetime.now(TAIPEI_TZ)

    event = {
        "source": source,
        "received_at_local": local_now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for key, value in (source_metadata or {}).items():
        if value is not None and value != "":
            event[key] = value

    return event


def _base_daily_record(
    record_type: Literal["history", "questionnaire"],
    basic: dict,
    extraction: dict,
    company_id: str,
    branch_id: str,
    coach_id: str,
    target_user_id: str,
    user_name: str,
    source: Literal["photo"],
    source_metadata: dict | None,
    warnings: list[str],
) -> dict:
    local_now = datetime.now(TAIPEI_TZ)
    unreadable = extraction.get("unreadable_or_ambiguous_items") or []

    return {
        "company_id": company_id,
        "branch_id": branch_id,
        "coach_id": coach_id,
        "used_id": target_user_id,
        "user_name": user_name,
        "type": record_type,
        "gemini_model": GEMINI_MODEL,
        "gemini_provider": "vertex_ai",

        # OCR 的基本資料保留，但不拿來改變 selected user 綁定。
        "gender": basic.get("gender"),
        "age": basic.get("age"),
        "profile": basic,

        # 同日不同來源統一累積。
        "sources": [source],
        "source_events": [
            _build_source_event(source, source_metadata)
        ],
        "upload_count": 1,


        "data_quality": {
            "review_required": bool(warnings or unreadable),
            "validation_warnings": warnings,
            "unreadable_or_ambiguous_items": unreadable,
            "merge_conflicts": [],
        },

        "record_date": local_now.strftime("%Y-%m-%d"),
        "recorded_at_local": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "last_updated_at_local": local_now.strftime("%Y-%m-%d %H:%M:%S"),
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }



def _boolean_storage_payload(profile: dict | None, allowed_labels: dict[str, str]) -> dict[str, bool]:
    """
    Firestore answer payload只保留固定 schema 的 bool。
    None 代表「本次沒看到/無法判斷」，不寫入，避免分頁補拍把舊值蓋掉。
    """
    payload: dict[str, bool] = {}
    for key, value in (profile or {}).items():
        if key in allowed_labels and isinstance(value, bool):
            payload[key] = value
    return payload


def _validate_exclusive_groups(
    profile: dict | None, groups: dict[str, list[str]]
) -> list[str]:
    warnings: list[str] = []
    profile = profile or {}
    for group_name, keys in groups.items():
        selected = [key for key in keys if profile.get(key) is True]
        if len(selected) > 1:
            warnings.append(
                f"單選題 {group_name} 同時辨識到多個選項：{', '.join(selected)}"
            )
    return warnings


def _selected_boolean_labels(profile: dict | None, labels: dict[str, str]) -> list[str]:
    profile = profile or {}
    return [label for key, label in labels.items() if profile.get(key) is True]


def _history_display_labels(profile: dict | None) -> list[str]:
    """
    前端健康/安全資訊的顯示文字。

    H01「目前是否有規律運動」是雙向狀態題：
    - True  -> 有規律運動
    - False -> 沒有規律運動
    - None  -> 本次未看到/無法判斷，不顯示

    其他 history 欄位維持既有規則：只有 True 才顯示其 label。
    這個 helper 只影響 API response 的 display，不改 Firestore history payload。
    """
    profile = profile or {}
    output: list[str] = []

    exercise_value = profile.get("exercises_regularly_currently")
    if exercise_value is True:
        output.append("有規律運動")
    elif exercise_value is False:
        output.append("沒有規律運動")

    for key, label in HISTORY_FIELD_LABELS.items():
        if key == "exercises_regularly_currently":
            continue
        if profile.get(key) is True:
            output.append(label)

    return output


def build_history_record(
    extraction: dict,
    company_id: str,
    branch_id: str,
    coach_id: str,
    target_user_id: str,
    user_name: str,
    source: Literal["photo"],
    source_metadata: dict | None = None,
) -> dict:
    basic, warnings = _validate_questionnaire_identity(
        extraction, target_user_id, user_name
    )

    record = _base_daily_record(
        record_type="history",
        basic=basic,
        extraction=extraction,
        company_id=company_id,
        branch_id=branch_id,
        coach_id=coach_id,
        target_user_id=target_user_id,
        user_name=user_name,
        source=source,
        source_metadata=source_metadata,
        warnings=warnings,
    )

    warnings.extend(
        _validate_exclusive_groups(
            extraction.get("health_profile"), HISTORY_EXCLUSIVE_BOOL_GROUPS
        )
    )
    if warnings:
        record["data_quality"]["review_required"] = True
        record["data_quality"]["validation_warnings"] = _merge_unique_lists(
            record["data_quality"].get("validation_warnings") or [], warnings
        )

    record["history_version"] = HISTORY_VERSION
    record["history"] = _boolean_storage_payload(
        extraction.get("health_profile"), HISTORY_FIELD_LABELS
    )

    return record


def build_lifestyle_questionnaire_record(
    extraction: dict,
    company_id: str,
    branch_id: str,
    coach_id: str,
    target_user_id: str,
    user_name: str,
    source: Literal["photo"],
    source_metadata: dict | None = None,
) -> dict:
    basic, warnings = _validate_questionnaire_identity(
        extraction, target_user_id, user_name
    )

    record = _base_daily_record(
        record_type="questionnaire",
        basic=basic,
        extraction=extraction,
        company_id=company_id,
        branch_id=branch_id,
        coach_id=coach_id,
        target_user_id=target_user_id,
        user_name=user_name,
        source=source,
        source_metadata=source_metadata,
        warnings=warnings,
    )

    # 單選題品質檢查；不改答案，只要求人工確認。
    warnings.extend(
        _validate_exclusive_groups(
            extraction.get("lifestyle_profile"), QUESTIONNAIRE_EXCLUSIVE_BOOL_GROUPS
        )
    )
    if warnings:
        record["data_quality"]["review_required"] = True
        record["data_quality"]["validation_warnings"] = _merge_unique_lists(
            record["data_quality"].get("validation_warnings") or [], warnings
        )

    record["questionnaire_version"] = QUESTIONNAIRE_VERSION
    record["questionnaire"] = _boolean_storage_payload(
        extraction.get("lifestyle_profile"), QUESTIONNAIRE_FIELD_LABELS
    )

    return record


def should_save_history(extraction: dict) -> bool:
    # 只要本次有任何可判定 bool（true 或 false）就可儲存；全 None 則不建檔。
    return bool(
        _boolean_storage_payload(extraction.get("health_profile"), HISTORY_FIELD_LABELS)
    )


def should_save_questionnaire(extraction: dict) -> bool:
    return bool(
        _boolean_storage_payload(
            extraction.get("lifestyle_profile"), QUESTIONNAIRE_FIELD_LABELS
        )
    )


def build_daily_doc_id(
    company_id: str,
    branch_id: str,
    coach_id: str,
    target_user_id: str,
    record_type: Literal["history", "questionnaire"],
    record_date: str | None = None,
) -> str:
    """
    同一人、同一天、同一 type 固定同一個 Document ID。

    例如：
    COMP_DrSafe_BR_NB_01_Coach_03_A001_history_20260814
    COMP_DrSafe_BR_NB_01_Coach_03_A001_questionnaire_20260814
    """
    if record_date:
        date_id = record_date.replace("-", "")
    else:
        date_id = datetime.now(TAIPEI_TZ).strftime("%Y%m%d")

    return "_".join([
        sanitize_doc_id_part(company_id, "COMPANY"),
        sanitize_doc_id_part(branch_id, "BRANCH"),
        sanitize_doc_id_part(coach_id, "COACH"),
        sanitize_doc_id_part(target_user_id, "USER"),
        record_type,
        date_id,
    ])


def _merge_data_quality(
    old_quality: dict | None,
    new_quality: dict | None,
    merge_conflicts: list[dict],
) -> dict:
    old_quality = old_quality or {}
    new_quality = new_quality or {}

    warnings = _merge_unique_lists(
        old_quality.get("validation_warnings") or [],
        new_quality.get("validation_warnings") or [],
    )

    unreadable = _merge_unique_lists(
        _drop_resolved_mapping_warnings(
            old_quality.get("unreadable_or_ambiguous_items") or []
        ),
        _drop_resolved_mapping_warnings(
            new_quality.get("unreadable_or_ambiguous_items") or []
        ),
    )

    old_conflicts = old_quality.get("merge_conflicts") or []
    all_conflicts = _merge_unique_lists(old_conflicts, merge_conflicts)

    return {
        "review_required": bool(
            old_quality.get("review_required")
            or new_quality.get("review_required")
            or warnings
            or unreadable
            or all_conflicts
        ),
        "validation_warnings": warnings,
        "unreadable_or_ambiguous_items": unreadable,
        "merge_conflicts": all_conflicts,
    }


def merge_daily_record(existing: dict, incoming: dict) -> dict:
    """
    合併同一天同一使用者同 type 的資料。

    metadata 的規則與內容欄位分開處理，避免 upload_count / timestamp
    被一般遞迴 merge 破壞。
    """
    if not existing:
        return incoming

    merged = dict(existing)
    conflicts: list[dict] = []

    content_keys = [
        "profile",
        "history",
        "questionnaire",
    ]

    for key in content_keys:
        if key in incoming:
            merged[key] = _smart_merge_values(
                existing.get(key),
                incoming.get(key),
                key,
                conflicts,
            )

    # 固定身份欄位以既有文件為主；不存在才補新值。
    for key in [
        "company_id",
        "branch_id",
        "coach_id",
        "used_id",
        "user_name",
        "type",
        "record_date",
        "recorded_at_local",
    ]:
        if _is_empty_value(merged.get(key)):
            merged[key] = incoming.get(key)

    # profile 合併後同步 top-level age / gender。
    merged_profile = merged.get("profile") or {}
    if not _is_empty_value(merged_profile.get("gender")):
        merged["gender"] = merged_profile.get("gender")
    elif _is_empty_value(merged.get("gender")):
        merged["gender"] = incoming.get("gender")

    if not _is_empty_value(merged_profile.get("age")):
        merged["age"] = merged_profile.get("age")
    elif _is_empty_value(merged.get("age")):
        merged["age"] = incoming.get("age")

    merged["sources"] = _merge_unique_lists(
        existing.get("sources") or [],
        incoming.get("sources") or [],
    )
    merged["source_events"] = _merge_unique_lists(
        existing.get("source_events") or [],
        incoming.get("source_events") or [],
    )

    merged["upload_count"] = int(existing.get("upload_count") or 0) + 1
    merged["gemini_model"] = incoming.get("gemini_model") or existing.get("gemini_model")

    if incoming.get("history_version"):
        merged["history_version"] = incoming["history_version"]
    if incoming.get("questionnaire_version"):
        merged["questionnaire_version"] = incoming["questionnaire_version"]

    merged["data_quality"] = _merge_data_quality(
        existing.get("data_quality"),
        incoming.get("data_quality"),
        conflicts,
    )

    # createdAt 保留原始值；updatedAt 每次補拍都更新。
    if "createdAt" not in merged:
        merged["createdAt"] = firestore.SERVER_TIMESTAMP

    local_now = datetime.now(TAIPEI_TZ)
    merged["last_updated_at_local"] = local_now.strftime("%Y-%m-%d %H:%M:%S")
    merged["updatedAt"] = firestore.SERVER_TIMESTAMP

    return merged


def upsert_daily_record(doc_id: str, incoming: dict) -> tuple[dict, bool]:
    """
    讀取當日既有 Document；存在則智慧合併，不存在才新增。

    回傳：
      merged_record, existed_before
    """
    doc_ref = app.state.db.collection("HealthRecords").document(doc_id)
    snapshot = doc_ref.get()

    existed_before = snapshot.exists
    existing = snapshot.to_dict() if existed_before else {}

    merged = merge_daily_record(existing or {}, incoming)
    doc_ref.set(merged)

    return merged, existed_before


def _public_daily_result(
    extraction: dict,
    history_record: dict | None,
    questionnaire_record: dict | None,
    history_doc_id: str | None,
    questionnaire_doc_id: str | None,
    source: str,
) -> dict:
    """
    前端使用的結果，不直接回傳 Firestore Timestamp sentinel。
    顯示的是「本次辨識 + 當日已合併資料」。
    """
    history_profile = (history_record or {}).get("profile") or {}
    questionnaire_profile = (questionnaire_record or {}).get("profile") or {}

    profile = (
        history_profile
        or questionnaire_profile
        or extraction.get("basic_info")
        or {}
    )

    history_quality = (history_record or {}).get("data_quality") or {}
    questionnaire_quality = (questionnaire_record or {}).get("data_quality") or {}

    warnings = _merge_unique_lists(
        history_quality.get("validation_warnings") or [],
        questionnaire_quality.get("validation_warnings") or [],
    )
    unreadable = _merge_unique_lists(
        history_quality.get("unreadable_or_ambiguous_items") or [],
        questionnaire_quality.get("unreadable_or_ambiguous_items") or [],
    )
    conflicts = _merge_unique_lists(
        history_quality.get("merge_conflicts") or [],
        questionnaire_quality.get("merge_conflicts") or [],
    )

    return {
        "source": source,
        "user_name": (
            (history_record or {}).get("user_name")
            or (questionnaire_record or {}).get("user_name")
        ),
        "profile": profile,

        "health_profile": (
            (history_record or {}).get("history")
            or extraction.get("health_profile")
            or {}
        ),
        "lifestyle_profile": (
            (questionnaire_record or {}).get("questionnaire")
            or extraction.get("lifestyle_profile")
            or {}
        ),

        # display 僅供前端閱讀，不寫入 Firebase answer payload。
        "display": {
            "history_selected_labels": _history_display_labels(
                (history_record or {}).get("history")
                or extraction.get("health_profile")
                or {}
            ),
            "questionnaire_selected_labels": _selected_boolean_labels(
                (questionnaire_record or {}).get("questionnaire")
                or extraction.get("lifestyle_profile")
                or {},
                QUESTIONNAIRE_FIELD_LABELS,
            ),
        },

        "saved_documents": {
            "history": {
                "doc_id": history_doc_id,
                "upload_count": (history_record or {}).get("upload_count"),
                "sources": (history_record or {}).get("sources") or [],
                "last_updated_at_local":
                    (history_record or {}).get("last_updated_at_local"),
            } if history_doc_id else None,

            "questionnaire": {
                "doc_id": questionnaire_doc_id,
                "upload_count": (questionnaire_record or {}).get("upload_count"),
                "sources": (questionnaire_record or {}).get("sources") or [],
                "last_updated_at_local":
                    (questionnaire_record or {}).get("last_updated_at_local"),
            } if questionnaire_doc_id else None,
        },

        "data_quality": {
            "review_required": bool(warnings or unreadable or conflicts),
            "validation_warnings": warnings,
            "unreadable_or_ambiguous_items": unreadable,
            "merge_conflicts": conflicts,
        },
    }


def save_questionnaire_extraction(
    extraction: dict,
    company_id: str,
    branch_id: str,
    coach_id: str,
    target_user_id: str,
    user_name: str,
    source: Literal["photo"],
    source_metadata: dict | None = None,
) -> dict:
    """
    一次 Gemini extraction 自動拆為：
    - history
    - questionnaire

    同日已存在時做 upsert + smart merge。
    """
    local_now = datetime.now(TAIPEI_TZ)
    record_date = local_now.strftime("%Y-%m-%d")

    history_doc_id = None
    questionnaire_doc_id = None
    history_record = None
    questionnaire_record = None
    history_existed = False
    questionnaire_existed = False

    if should_save_history(extraction):
        incoming_history = build_history_record(
            extraction=extraction,
            company_id=company_id,
            branch_id=branch_id,
            coach_id=coach_id,
            target_user_id=target_user_id,
            user_name=user_name,
            source=source,
            source_metadata=source_metadata,
        )

        history_doc_id = build_daily_doc_id(
            company_id,
            branch_id,
            coach_id,
            target_user_id,
            "history",
            record_date,
        )

        history_record, history_existed = upsert_daily_record(
            history_doc_id, incoming_history
        )

    if should_save_questionnaire(extraction):
        incoming_questionnaire = build_lifestyle_questionnaire_record(
            extraction=extraction,
            company_id=company_id,
            branch_id=branch_id,
            coach_id=coach_id,
            target_user_id=target_user_id,
            user_name=user_name,
            source=source,
            source_metadata=source_metadata,
        )

        questionnaire_doc_id = build_daily_doc_id(
            company_id,
            branch_id,
            coach_id,
            target_user_id,
            "questionnaire",
            record_date,
        )

        questionnaire_record, questionnaire_existed = upsert_daily_record(
            questionnaire_doc_id, incoming_questionnaire
        )

    if not history_doc_id and not questionnaire_doc_id:
        raise HTTPException(
            status_code=422,
            detail="未辨識到可儲存的疾病/手術史或生活/運動問卷資訊，請確認照片清晰度",
        )

    return {
        "history_doc_id": history_doc_id,
        "questionnaire_doc_id": questionnaire_doc_id,
        "history_existed_before": history_existed,
        "questionnaire_existed_before": questionnaire_existed,
        "ai_result": _public_daily_result(
            extraction=extraction,
            history_record=history_record,
            questionnaire_record=questionnaire_record,
            history_doc_id=history_doc_id,
            questionnaire_doc_id=questionnaire_doc_id,
            source=source,
        ),
    }



# -----------------------------------------------------------------------------
# 8. API
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 8A. Muzili 專屬身體組成 API：兩張固定圖片 + 人工身高/年齡/性別
# -----------------------------------------------------------------------------
@app.post("/api/analyze-muzili-body-composition")
async def analyze_muzili_body_composition(
    body_data_file: UploadFile = File(...),
    segmental_muscle_file: UploadFile = File(...),
    height_cm: float = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    target_user_id: str = Form(...),
    user_name: str = Form(...),
    company_id: str = Form(...),
    branch_id: str = Form(...),
    coach_id: str = Form(...),
    actor: ScannerActor = Depends(require_scanner_actor),
):
    resolved_user_id = require_context_id(target_user_id, "target_user_id")
    resolved_company_id, resolved_branch_id, resolved_coach_id, student_data = (
        _validate_scanner_context(
            actor=actor,
            company_id=company_id,
            branch_id=branch_id,
            coach_id=coach_id,
            used_id=resolved_user_id,
        )
    )
    resolved_user_name = (
        _clean_id(student_data.get("user_name"))
        or _clean_id(user_name)
        or resolved_user_id
    )

    # 人工欄位防呆。這三個欄位不交給 AI 判斷。
    if not 100.0 <= float(height_cm) <= 250.0:
        raise HTTPException(status_code=422, detail="身高必須介於 100～250 cm")

    if not 1 <= int(age) <= 120:
        raise HTTPException(status_code=422, detail="年齡必須介於 1～120 歲")

    clean_gender = str(gender or "").strip().lower()
    if clean_gender not in {"male", "female"}:
        raise HTTPException(status_code=422, detail="gender 必須為 male 或 female")

    try:
        # 兩張圖片各自具名，避免 files[0] / files[1] 順序混淆。
        body_bytes = await body_data_file.read()
        segmental_bytes = await segmental_muscle_file.read()

        if not body_bytes:
            raise HTTPException(status_code=422, detail="缺少 Muzili 身體數據圖片")
        if not segmental_bytes:
            raise HTTPException(status_code=422, detail="缺少 Muzili 節段肌肉分析圖片")

        try:
            body_image = Image.open(io.BytesIO(body_bytes))
            body_image = preprocess_image(body_image, max_size=(1800, 1800))
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="無法讀取 Muzili 身體數據圖片",
            ) from exc

        try:
            segmental_image = Image.open(io.BytesIO(segmental_bytes))
            segmental_image = preprocess_image(segmental_image, max_size=(1800, 1800))
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail="無法讀取 Muzili 節段肌肉分析圖片",
            ) from exc

        raw_result = await asyncio.to_thread(
            extract_muzili_health_data,
            body_image,
            segmental_image,
            {
                "company_id": resolved_company_id,
                "branch_id": resolved_branch_id,
                "coach_id": resolved_coach_id,
                "used_id": resolved_user_id,
            },
        )

        # 人工資料覆蓋 AI；來源明確且可追溯。
        raw_result["device_type"] = "other"
        raw_result["height_m"] = round(float(height_cm) / 100.0, 3)
        raw_result["age"] = int(age)
        raw_result["gender"] = clean_gender

        # Muzili 顯示 FFMI，不是 CoachOS 使用的 SMI。
        # SMI 由四肢肌肉量 ALM / height^2 在既有 calculate_derived_metrics() 計算。
        raw_result["smi_kg_m2"] = None
        raw_result["inbody_score"] = None

        result, validation_warnings = validate_health_data(raw_result)

        # Muzili 專屬完整性檢查：兩張固定頁面應提供全部必要欄位。
        # 若第二張誤上傳「脂肪量」頁，四肢肌肉欄位會缺失並直接拒絕存檔。
        required_fields = {
            "weight_kg": "體重",
            "body_fat_percentage": "體脂率",
            "body_fat_mass_kg": "脂肪量",
            "fat_free_mass_kg": "去脂體重",
            "basal_metabolic_rate": "基礎代謝",
            "visceral_fat_value": "內臟脂肪",
            "segmental_muscle_la": "左臂肌肉量",
            "segmental_muscle_ra": "右臂肌肉量",
            "segmental_muscle_ll": "左腿肌肉量",
            "segmental_muscle_rl": "右腿肌肉量",
        }
        missing_fields = [
            label
            for key, label in required_fields.items()
            if result.get(key) is None
        ]
        if missing_fields:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Muzili 圖片資料不完整，請確認上傳頁面是否正確。缺少："
                    + "、".join(missing_fields)
                ),
            )

        firebase_data = build_firebase_data(
            result=result,
            company_id=resolved_company_id,
            branch_id=resolved_branch_id,
            coach_id=resolved_coach_id,
            target_user_id=resolved_user_id,
            user_name=resolved_user_name,
            validation_warnings=validation_warnings,
        )

        # Provenance：保留既有 body_composition schema，只加來源欄位。
        firebase_data["source_system"] = "muzili"
        firebase_data["input_sources"] = {
            "height_m": "manual",
            "age": "manual",
            "gender": "manual",
            "body_composition": "muzili_body_data_image",
            "segmental_muscle": "muzili_segmental_muscle_image",
        }

        local_now = datetime.now(TAIPEI_TZ)
        timestamp_id = local_now.strftime("%Y%m%d_%H%M%S")
        doc_id = "_".join([
            sanitize_doc_id_part(resolved_company_id, "COMPANY"),
            sanitize_doc_id_part(resolved_branch_id, "BRANCH"),
            sanitize_doc_id_part(resolved_coach_id, "COACH"),
            sanitize_doc_id_part(resolved_user_id, "USER"),
            "bodycomp",
            "muzili",
            timestamp_id,
        ])

        app.state.db.collection("HealthRecords").document(doc_id).set(firebase_data)

        response_data = dict(firebase_data)
        response_data["createdAt"] = local_now.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success",
            "message": "Muzili 雙圖解析、CoachOS 評分與存檔成功",
            "doc_id": doc_id,
            "ai_result": response_data,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Muzili API 執行錯誤")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze-inbody")  # 舊 endpoint 保留，避免既有前端立即失效
@app.post("/api/analyze-body-composition")
async def analyze_body_composition(
    file: UploadFile = File(...),
    target_user_id: str = Form(...),
    user_name: str = Form(...),
    company_id: str = Form(...),
    branch_id: str = Form(...),
    coach_id: str = Form(...),
    actor: ScannerActor = Depends(require_scanner_actor),
):
    resolved_user_id = require_context_id(target_user_id, "target_user_id")
    resolved_company_id, resolved_branch_id, resolved_coach_id, student_data = (
        _validate_scanner_context(
            actor=actor,
            company_id=company_id,
            branch_id=branch_id,
            coach_id=coach_id,
            used_id=resolved_user_id,
        )
    )
    resolved_user_name = (
        _clean_id(student_data.get("user_name"))
        or _clean_id(user_name)
        or resolved_user_id
    )

    try:
        image_bytes = await file.read()
        raw_img = Image.open(io.BytesIO(image_bytes))
        processed_img = preprocess_image(raw_img)

        # gender 與所有 body-composition measurement 一樣，直接從這張報表 OCR 取得。
        raw_result = await asyncio.to_thread(
            extract_health_data,
            processed_img,
            {
                "company_id": resolved_company_id,
                "branch_id": resolved_branch_id,
                "coach_id": resolved_coach_id,
                "used_id": resolved_user_id,
            },
        )
        result, validation_warnings = validate_health_data(raw_result)

        firebase_data = build_firebase_data(
            result=result,
            company_id=resolved_company_id,
            branch_id=resolved_branch_id,
            coach_id=resolved_coach_id,
            target_user_id=resolved_user_id,
            user_name=resolved_user_name,
            validation_warnings=validation_warnings,
        )

        # Document ID 對齊 upload_manager.py 的前綴格式：
        # company_branch_coach_usedid_bodycomp_device_YYYYMMDD_HHMMSS
        local_now = datetime.now(TAIPEI_TZ)
        timestamp_id = local_now.strftime("%Y%m%d_%H%M%S")
        doc_id = "_".join([
            sanitize_doc_id_part(resolved_company_id, "COMPANY"),
            sanitize_doc_id_part(resolved_branch_id, "BRANCH"),
            sanitize_doc_id_part(resolved_coach_id, "COACH"),
            sanitize_doc_id_part(resolved_user_id, "USER"),
            "bodycomp",
            sanitize_doc_id_part(result.get("device_type"), "unknown"),
            timestamp_id,
        ])

        app.state.db.collection("HealthRecords").document(doc_id).set(firebase_data)

        response_data = dict(firebase_data)
        response_data["createdAt"] = local_now.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success",
            "message": "雲端解析、CoachOS 評分與存檔成功",
            "doc_id": doc_id,
            "ai_result": response_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API 執行錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 8B. CoachOS 標準問卷 API：手機多頁拍照
# -----------------------------------------------------------------------------
@app.post("/api/analyze-questionnaire")
async def analyze_questionnaire(
    files: list[UploadFile] = File(...),
    target_user_id: str = Form(...),
    user_name: str = Form(...),
    company_id: str = Form(...),
    branch_id: str = Form(...),
    coach_id: str = Form(...),
    page_numbers: str = Form(""),
    actor: ScannerActor = Depends(require_scanner_actor),
):
    resolved_user_id = require_context_id(target_user_id, "target_user_id")
    resolved_company_id, resolved_branch_id, resolved_coach_id, student_data = (
        _validate_scanner_context(
            actor=actor,
            company_id=company_id,
            branch_id=branch_id,
            coach_id=coach_id,
            used_id=resolved_user_id,
        )
    )
    resolved_user_name = (
        _clean_id(student_data.get("user_name"))
        or _clean_id(user_name)
        or resolved_user_id
    )

    if not files:
        raise HTTPException(status_code=422, detail="至少需要一張 CoachOS 標準問卷照片")
    if len(files) > 4:
        raise HTTPException(status_code=422, detail="單次最多上傳 4 張問卷照片")

    try:
        processed_images: list[Image.Image] = []

        parsed_page_numbers: list[int] = []
        if page_numbers:
            try:
                raw_page_numbers = json.loads(page_numbers)
                if isinstance(raw_page_numbers, list):
                    parsed_page_numbers = [
                        int(value)
                        for value in raw_page_numbers
                        if isinstance(value, (int, float, str))
                        and str(value).strip().isdigit()
                        and 1 <= int(value) <= 4
                    ]
            except Exception:
                logger.warning(
                    f"問卷 page_numbers 無法解析，將只依圖片順序處理: {page_numbers!r}"
                )

        for uploaded in files:
            image_bytes = await uploaded.read()
            if not image_bytes:
                continue
            try:
                raw_img = Image.open(io.BytesIO(image_bytes))
                # 問卷整頁文字比 BIA 報表更密，保留較高解析度避免小字被縮掉。
                processed_images.append(
                    preprocess_image(raw_img, max_size=(1800, 1800))
                )
            except Exception:
                raise HTTPException(
                    status_code=422,
                    detail=f"無法讀取問卷圖片：{uploaded.filename or 'unknown'}",
                )

        if not processed_images:
            raise HTTPException(status_code=422, detail="沒有可解析的問卷圖片")

        logger.info(
            f"CoachOS 問卷 slot upload: user={resolved_user_id}, "
            f"page_slots={parsed_page_numbers or 'legacy-order'}, "
            f"image_count={len(processed_images)}"
        )

        extraction = await asyncio.to_thread(
            extract_questionnaire_from_images,
            processed_images,
            {
                "company_id": resolved_company_id,
                "branch_id": resolved_branch_id,
                "coach_id": resolved_coach_id,
                "used_id": resolved_user_id,
            },
        )

        saved = save_questionnaire_extraction(
            extraction=extraction,
            company_id=resolved_company_id,
            branch_id=resolved_branch_id,
            coach_id=resolved_coach_id,
            target_user_id=resolved_user_id,
            user_name=resolved_user_name,
            source="photo",
            source_metadata={
                "page_count": len(processed_images),
                "page_slots": (
                    parsed_page_numbers
                    if len(parsed_page_numbers) == len(processed_images)
                    else []
                ),
            },
        )

        created_or_updated = []
        if saved.get("history_doc_id"):
            action = (
                "updated"
                if saved.get("history_existed_before")
                else "created"
            )
            created_or_updated.append(f"history:{action}")

        if saved.get("questionnaire_doc_id"):
            action = (
                "updated"
                if saved.get("questionnaire_existed_before")
                else "created"
            )
            created_or_updated.append(f"questionnaire:{action}")

        return {
            "status": "success",
            "message": "CoachOS 標準問卷辨識成功；已依內容拆分並整合到當日 HealthRecords",
            "documents": {
                "history": saved.get("history_doc_id"),
                "questionnaire": saved.get("questionnaire_doc_id"),
            },
            "merge_actions": created_or_updated,
            "ai_result": saved["ai_result"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"問卷 API 執行錯誤: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# 9. Scanner identity / branch / member list
# -----------------------------------------------------------------------------
@app.get("/api/me")
async def scanner_me(
    actor: ScannerActor = Depends(require_scanner_actor),
):
    return {"status": "success", "actor": actor.model_dump()}


@app.get("/api/branches")
async def get_scanner_branches(
    actor: ScannerActor = Depends(require_scanner_actor),
):
    try:
        if actor.role == "owner":
            refs = (
                app.state.db.collection("Branches")
                .where("company_id", "==", actor.company_id)
                .stream()
            )
        else:
            snap = app.state.db.collection("Branches").document(actor.branch_id).get()
            refs = [snap] if snap.exists else []

        branches = []
        for snap in refs:
            data = snap.to_dict() or {}
            if _clean_id(data.get("company_id")) != actor.company_id:
                continue
            branches.append(
                {
                    "id": snap.id,
                    "name": data.get("name", snap.id),
                    "company_id": actor.company_id,
                }
            )

        branches.sort(key=lambda x: (str(x.get("name") or ""), x["id"]))
        return {"status": "success", "branches": branches}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("獲取 Scanner 分店名單失敗: %s", exc)
        raise HTTPException(status_code=500, detail="無法讀取分店名單") from exc


@app.get("/api/users")
async def get_users(
    branch_id: str,
    company_id: str | None = None,
    coach_id: str | None = None,
    actor: ScannerActor = Depends(require_scanner_actor),
):
    """
    Scanner member list always follows coach property:
      company + branch + coach_id.

    Manager/owner broader read scope remains exclusively in Coach API.
    """
    try:
        if company_id is not None and _clean_id(company_id) != actor.company_id:
            raise HTTPException(status_code=403, detail="company_id 與登入帳號不一致")
        if coach_id is not None and _clean_id(coach_id) != actor.coach_id:
            raise HTTPException(status_code=403, detail="coach_id 與登入帳號不一致")

        resolved_branch = _resolve_upload_branch(actor, branch_id)

        # Keep the Firestore query backward-compatible and simple:
        # query only by company + branch, then filter coach_id in Python.
        # This avoids making Scanner availability depend on a 3-field
        # compound query/index configuration and still preserves the exact
        # authorization rule: company + branch + coach_id.
        refs = (
            app.state.db.collection("Students")
            .where("company_id", "==", actor.company_id)
            .where("branch_id", "==", resolved_branch)
            .stream()
        )

        users_by_id: dict[str, dict] = {}
        for doc in refs:
            data = doc.to_dict() or {}

            # Only this login's coach property may appear in Scanner.
            if _clean_id(data.get("coach_id")) != actor.coach_id:
                continue

            used_id = _clean_id(data.get("used_id")) or doc.id
            if not used_id:
                continue

            users_by_id.setdefault(
                used_id,
                {
                    "id": used_id,
                    "name": data.get("user_name", "未提供名稱"),
                    "branch_id": resolved_branch,
                    "coach_id": actor.coach_id,
                },
            )

        users_list = list(users_by_id.values())
        users_list.sort(key=lambda x: (str(x.get("name") or ""), x["id"]))

        return {
            "status": "success",
            "branch_id": resolved_branch,
            "coach_id": actor.coach_id,
            "users": users_list,
        }

    except HTTPException:
        raise
    except Exception as exc:
        # logger.exception keeps the full traceback in Render logs.
        logger.exception("獲取雲端名單失敗")
        raise HTTPException(
            status_code=500,
            detail=f"無法讀取雲端學員名單（{type(exc).__name__}）",
        ) from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
