import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import html2canvas from 'html2canvas'
import { jsPDF } from 'jspdf'
import Muscle3DViewer, { renderMuscleReportSnapshots } from './Muscle3DViewer'
import './CoachDashboard.css'

const FIREBASE_API_KEY = import.meta.env.VITE_FIREBASE_API_KEY || ''
const COACH_API_URL = import.meta.env.VITE_COACH_API_URL || 'http://127.0.0.1:8001'
const STORAGE_KEY = 'coachosFirebaseSessionV5'
const EMPTY_EXERCISE_MESSAGE = '今日身體狀況沒有適合動作，請確認學員身體狀況'
const REPORT_LAYOUT_VERSION = 'muscle3d-v1-20260919'

const ISSUE_TYPE_LABELS = {
  dynamic: '動態評估',
  static: '靜態評估',
}

const ISSUE_LABELS = {
  ankle_mobility: '踝關節活動度受限',
  arms_fall_forward: '雙臂向前落',
  dorsiflexion: '踝背屈',
  elbow_adduction: '手肘向軀幹靠攏',
  elbow_flexion: '手肘彎曲',
  elbow_hyperextension: '手肘過度伸直',
  elbow_offground: '手肘離地',
  elbow_stability: '手肘穩定度不佳',
  excessive_forward_lean: '身體過度前傾',
  foot_eversion: '足外翻',
  foot_inversion: '足內翻',
  foot_lift: '足跟抬起離地',
  forward_head_posture: '頭前伸',
  head_retraction: '頭後縮',
  hip_extension: '髖部未充分彎曲',
  hip_flexion: '髖部未充分伸展',
  hip_mobility: '髖關節活動度受限',
  knee_extension: '膝蓋未充分彎曲',
  knee_falling_over_toe: '膝蓋過度向前',
  knee_flexion: '膝蓋未充分伸展',
  knee_hyperextension: '膝蓋過度伸展',
  leg_external_rotation: '腿部向外旋轉',
  leg_internal_rotation: '腿部向內旋轉',
  low_back_arches: '腰部過度伸展',
  lumbar_hyperextension: '腰部過度伸展',
  neck_hyperextension: '頸伸展',
  neck_hyperflexion: '頸屈曲',
  pelvic_obliquity: '骨盆傾斜',
  pelvic_rotation: '骨盆旋轉',
  pelvic_stability: '骨盆穩定度不佳',
  plantarflexion: '踝蹠屈',
  rib_hyperextension: '胸廓伸展',
  rib_hyperflexion: '胸廓屈曲',
  rib_rotation: '胸廓旋轉',
  ribcage_stability: '胸廓穩定度不佳',
  rounded_shoulders: '肩前引',
  scapula_stability: '肩胛穩定度不佳',
  scoliosis: '軀幹側向偏移',
  shoulder_depression: '肩膀過度下壓',
  shoulder_elevation: '肩膀過度上提',
  shoulder_extension: '手臂過度向後伸展',
  shoulder_flexion: '手臂過度向前抬起',
  shoulder_retraction: '肩胛內收',
  spine_mobility: '脊柱活動度受限',
  squat_too_deep: '下蹲過深',
  thoracic_mobility: '胸廓活動度受限',
  uneven_shoulders: '高低肩',
  bow_legged: 'O型腿',
  internally_rotated_shoulders: '含胸',
  knock_knees: 'X型腿',
}

function getIssueDisplayLabel(row = {}) {
  let issueType = String(row.issue_type || '').trim()
  let issueKey = String(row.issue_key || '').trim()
  const rawProblem = String(row.issue_problem || '').trim()

  // 相容舊版 API 僅回傳「dynamic - issue_key」的資料格式。
  if ((!issueType || !issueKey) && rawProblem) {
    const [rawType, ...rawKeyParts] = rawProblem.split(' - ')
    if (!issueType) issueType = String(rawType || '').trim()
    if (!issueKey) issueKey = rawKeyParts.join(' - ').trim()
  }

  const typeLabel = ISSUE_TYPE_LABELS[issueType] || issueType
  const issueLabel = ISSUE_LABELS[issueKey] || issueKey
  if (typeLabel && issueLabel) return `${typeLabel}｜${issueLabel}`
  return issueLabel || typeLabel || rawProblem || '--'
}

const INTENSITY_OPTIONS = [
  { key: 'light', label: '疲勞', caption: '今天放慢一點' },
  { key: 'moderate', label: '正常', caption: '照原節奏進行' },
  { key: 'vigorous', label: '加強鍛鍊', caption: '今天狀態不錯' },
]

const TRAINING_INTENSITY_LABELS = {
  low_to_moderate: '輕至中強度',
  moderate_to_high: '中至高強度',
  high: '高強度',
}

const ASSESSMENT_TABS = [
  ['score', '分數'],
  ['trend', '趨勢'],
  ['bodycomp', '體組成'],
  ['front', '正面姿態'],
  ['side', '側面姿態'],
  ['bridge', '橋式'],
  ['ohs', '過頭深蹲'],
  ['bird', '鳥狗式'],
  ['gait', '步態'],
]

const BODYCOMP_LABELS = {
  height_m: '身高',
  weight_kg: '體重',
  body_fat_percentage: '體脂率',
  body_fat_reference_low_pct: '體脂參考下限',
  body_fat_reference_high_pct: '體脂參考上限',
  body_fat_mass_kg: '脂肪量',
  basal_metabolic_rate: '基礎代謝',
  smi_kg_m2: '報表 SMI',
  visceral_fat_value: '內臟脂肪',
}

const DERIVED_LABELS = {
  BMI: 'BMI',
  ALM: 'ALM',
  SMI: 'SMI',
  AI_upper_pct: '上肢 AI',
  AI_lower_pct: '下肢 AI',
}

function saveSession(session) {
  if (session) localStorage.setItem(STORAGE_KEY, JSON.stringify(session))
  else localStorage.removeItem(STORAGE_KEY)
}

function loadStoredSession() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || null
  } catch {
    return null
  }
}

async function firebaseSignIn(email, password) {
  if (!FIREBASE_API_KEY) throw new Error('請先設定 VITE_FIREBASE_API_KEY')
  const response = await axios.post(
    `https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=${FIREBASE_API_KEY}`,
    { email, password, returnSecureToken: true },
  )
  const data = response.data
  return {
    idToken: data.idToken,
    refreshToken: data.refreshToken,
    email: data.email,
    localId: data.localId,
    expiresAt: Date.now() + Number(data.expiresIn || 3600) * 1000,
  }
}

async function refreshFirebaseSession(session) {
  if (!session?.refreshToken) throw new Error('登入資訊已失效')
  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    refresh_token: session.refreshToken,
  })
  const response = await axios.post(
    `https://securetoken.googleapis.com/v1/token?key=${FIREBASE_API_KEY}`,
    body,
    { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } },
  )
  const data = response.data
  return {
    ...session,
    idToken: data.id_token,
    refreshToken: data.refresh_token || session.refreshToken,
    localId: data.user_id || session.localId,
    expiresAt: Date.now() + Number(data.expires_in || 3600) * 1000,
  }
}

async function ensureFreshSession(session, onUpdate) {
  if (!session) throw new Error('尚未登入')
  if (session.expiresAt && session.expiresAt - Date.now() > 90_000) return session
  const refreshed = await refreshFirebaseSession(session)
  saveSession(refreshed)
  onUpdate?.(refreshed)
  return refreshed
}

async function coachRequest(session, onSessionUpdate, config) {
  const fresh = await ensureFreshSession(session, onSessionUpdate)
  return axios({
    baseURL: COACH_API_URL,
    ...config,
    headers: {
      ...(config.headers || {}),
      Authorization: `Bearer ${fresh.idToken}`,
    },
  })
}

function safeNumber(value, decimals = 1) {
  if (value === null || value === undefined || value === '') return '--'
  const n = Number(value)
  if (!Number.isFinite(n)) return String(value)
  if (decimals === 0) return String(Math.round(n))
  return n.toFixed(decimals)
}

function formatAssessmentValue(metric, value) {
  if (value === null || value === undefined || value === '') return '--'
  const number = safeNumber(value, metric.decimals ?? 1)
  const unit = metric.unit || ''
  if (!unit) return number
  if (unit === '°' || unit === '%') return `${number}${unit}`
  return `${number} ${unit}`
}

function formatBodyValue(key, value) {
  if (value === null || value === undefined || value === '') return '--'
  const n = Number(value)
  const display = Number.isFinite(n) ? (Number.isInteger(n) ? n : n.toFixed(1)) : value
  if (key.endsWith('_percentage') || key.endsWith('_pct')) return `${display}%`
  if (key === 'height_m') return `${display} m`
  if (key === 'weight_kg' || key.endsWith('_mass_kg')) return `${display} kg`
  if (key === 'basal_metabolic_rate') return `${display} kcal`
  return display
}

function initials(name = '') {
  const compact = name.trim()
  if (!compact) return 'CO'
  if (/^[\u4e00-\u9fff]/.test(compact)) return compact.slice(-2)
  return compact.split(/\s+/).slice(0, 2).map((x) => x[0]?.toUpperCase()).join('')
}

function roleLabel(role) {
  return { coach: 'Coach', manager: 'Manager', owner: 'Owner' }[role] || role
}

function taipeiDateParts(date = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Taipei',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date)
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]))
  const year = values.year || ''
  const month = values.month || ''
  const day = values.day || ''
  return {
    display: `${year}/${month}/${day}`,
    compact: `${year}${month}${day}`,
    iso: `${year}-${month}-${day}`,
  }
}

function reportGenderLabel(value) {
  const gender = String(value || '').trim().toLowerCase()
  if (gender === 'm' || gender === 'male') return '男性'
  if (gender === 'f' || gender === 'female') return '女性'
  return '待資料收集'
}

function reportIntensityLabel(value) {
  return INTENSITY_OPTIONS.find((item) => item.key === value)?.label || '正常'
}

function trainingIntensityLabel(value) {
  const key = String(value || '').trim()
  if (!key) return '--'
  return TRAINING_INTENSITY_LABELS[key] || key
}

function reportStateLabel(state) {
  if (state === 'ready') return '資料齊全'
  if (state === 'partial') return '部分資料'
  return '待資料收集'
}

function formatTrendDateLabel(value) {
  const text = String(value || '--').trim()
  const match = text.match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (match) return `${match[2]}/${match[3]}`
  return text
}

function sanitizeReportFilenamePart(value) {
  return String(value || 'member').trim().replace(/[\\/:*?"<>|]+/g, '-').replace(/\s+/g, '_') || 'member'
}

function ReportPageFooter({ page, total = 6 }) {
  return (
    <div className="member-report-footer">
      <span>CoachOS · 維特魯威運動科技｜運動與健康促進參考，非醫療診斷</span>
      <span>{page} / {total}</span>
    </div>
  )
}

function ReportTrendChart({ title, rows = [] }) {
  if (!rows.length) {
    return (
      <section className="report-card report-trend-card">
        <h3>{title}</h3>
        <div className="report-empty">目前尚無可用的趨勢資料</div>
      </section>
    )
  }

  // v5: labels are regular HTML, not SVG <text>.
  // html2canvas/jsPDF scales SVG text together with its viewBox, which made
  // the previous font-size changes visually ineffective in the final PDF.
  // HTML labels keep their CSS px size before the whole A4 page is rasterized.
  const width = 1000
  const height = 300
  const padX = 88
  const padTop = 70
  const padBottom = 68
  const numericRows = rows
    .map((row) => ({ ...row, n: Number(row.value) }))
    .filter((row) => Number.isFinite(row.n))

  if (!numericRows.length) {
    return (
      <section className="report-card report-trend-card">
        <h3>{title}</h3>
        <div className="report-empty">目前尚無可用的趨勢資料</div>
      </section>
    )
  }

  const values = numericRows.map((row) => row.n)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = numericRows.map((row, index) => {
    const x = numericRows.length === 1
      ? width / 2
      : padX + (index * (width - padX * 2)) / (numericRows.length - 1)
    const y = height - padBottom - ((row.n - min) / range) * (height - padTop - padBottom)
    return {
      ...row,
      x,
      y,
      leftPct: (x / width) * 100,
      topPct: (y / height) * 100,
    }
  })

  return (
    <section className="report-card report-trend-card" data-report-chart-version={REPORT_LAYOUT_VERSION}>
      <div className="report-section-heading">
        <h3>{title}</h3>
        <span>{rows.length >= 2 ? '最近五次趨勢' : '目前僅一筆資料'}</span>
      </div>

      <div className="report-trend-plot">
        <svg
          className="report-trend-svg"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <line
            x1={padX}
            y1={height - padBottom}
            x2={width - padX}
            y2={height - padBottom}
            className="report-chart-axis"
          />
          {points.length > 1 && (
            <polyline
              points={points.map((p) => `${p.x},${p.y}`).join(' ')}
              className="report-chart-line"
            />
          )}
          {points.map((point, index) => (
            <circle
              key={`dot-${point.date || 'date'}-${index}`}
              cx={point.x}
              cy={point.y}
              r="9"
              className="report-chart-dot"
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>

        {points.map((point, index) => (
          <span
            key={`value-${point.date || 'date'}-${index}`}
            className="report-chart-value-html"
            style={{ left: `${point.leftPct}%`, top: `${point.topPct}%` }}
          >
            {safeNumber(point.value, 1)}
          </span>
        ))}

        {points.map((point, index) => (
          <span
            key={`date-${point.date || 'date'}-${index}`}
            className="report-chart-date-html"
            style={{ left: `${point.leftPct}%` }}
          >
            {formatTrendDateLabel(point.date)}
          </span>
        ))}
      </div>

      {rows.length === 1 && (
        <div className="report-inline-note">目前僅有一次評估資料，尚無法判讀前後趨勢。</div>
      )}
    </section>
  )
}

function ReportMetricSection({ page, title }) {
  const dates = page?.dates || []
  const metrics = page?.metrics || []
  const resolvedTitle = title || page?.title || '評估項目'

  if (!dates.length || !metrics.length) {
    return (
      <section className="report-card report-metric-section">
        <h3>{resolvedTitle}</h3>
        <div className="report-empty">目前尚無可用資料</div>
      </section>
    )
  }

  return (
    <section className="report-card report-metric-section">
      <div className="report-section-heading">
        <h3>{resolvedTitle}</h3>
        <span>{dates.length >= 2 ? '最近兩次比較' : '目前僅一次評估'}</span>
      </div>
      {dates.length === 1 && <div className="report-inline-note">目前僅有一次評估資料，尚無法進行前後比較。</div>}
      <div className="report-table">
        <div className="report-table-row report-table-head" style={{ '--report-result-cols': dates.length }}>
          <div>評估指標</div>
          {dates.map((date, index) => <div key={`${date}-${index}`}>{date || '--'}</div>)}
          <div>參考</div>
        </div>
        {metrics.map((metric) => (
          <div className="report-table-row" style={{ '--report-result-cols': dates.length }} key={metric.key}>
            <div className="report-table-label">{metric.label}</div>
            {dates.map((_, index) => (
              <div className={metric.abnormal_values?.[index] === true ? 'report-value-abnormal' : ''} key={index}>
                {formatAssessmentValue(metric, metric.values?.[index])}
              </div>
            ))}
            <div className="report-reference">{metric.reference_label || '參考'} {metric.reference || '-'}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function ReportBodyComp({ data }) {
  if (!data) {
    return (
      <section className="report-card">
        <h3>Body Composition | 體組成分析</h3>
        <div className="report-empty">目前尚無 Body Composition 資料</div>
      </section>
    )
  }

  const measurements = data.measurements || {}
  const derived = data.derived_metrics || {}
  const segmental = data.segmental_muscle || {}
  const comparisons = data.comparisons || {}
  const hidden = new Set(['body_fat_reference_low_pct', 'body_fat_reference_high_pct', 'fat_free_mass_kg'])
  const measurementRows = Object.entries(measurements).filter(([key]) => !hidden.has(key))
  const derivedRows = Object.entries(derived)

  const renderBodyItem = (scope, key, value) => {
    const comparison = comparisons[`${scope}.${key}`]
    const abnormal = comparison?.abnormal === true
    const label = scope === 'measurements' ? (BODYCOMP_LABELS[key] || key) : (DERIVED_LABELS[key] || key)
    const formatted = scope === 'measurements'
      ? formatBodyValue(key, value)
      : (key.includes('pct') ? `${safeNumber(value, 1)}%` : safeNumber(value, 1))
    return (
      <div className={`report-data-tile ${abnormal ? 'abnormal' : ''}`} key={`${scope}-${key}`}>
        <span>{label}</span>
        <strong>{formatted}</strong>
        {comparison?.reference && <small>參考 {comparison.reference}</small>}
      </div>
    )
  }

  return (
    <section className="report-card report-bodycomp-card">
      <div className="report-section-heading">
        <h3>Body Composition | 體組成分析</h3>
        <span>{data.date || '--'} · {(data.device_type || '').toUpperCase()}</span>
      </div>

      <div className="report-body-score-row">
        <div className="report-score-orbit">
          <span>{data.scores?.coachos?.value ?? '--'}</span>
          <small>CoachOS 體組成分數</small>
        </div>
        <div className="report-body-meta">
          <div><span>年齡</span><strong>{data.age ?? '--'}</strong></div>
          <div><span>性別</span><strong>{reportGenderLabel(data.gender)}</strong></div>
          <div><span>設備</span><strong>{(data.device_type || '--').toUpperCase()}</strong></div>
        </div>
      </div>

      <div className="report-subtitle">身體組成與衍生指標</div>
      <div className="report-data-grid">
        {measurementRows.map(([key, value]) => renderBodyItem('measurements', key, value))}
        {derivedRows.map(([key, value]) => renderBodyItem('derived_metrics', key, value))}
      </div>

      <div className="report-subtitle">節段肌肉量</div>
      <div className="report-segment-grid">
        <div><span>右上肢</span><strong>{safeNumber(segmental.ra ?? segmental.right_arm_kg ?? segmental.ra_kg ?? segmental.segmental_muscle_ra, 2)} kg</strong></div>
        <div><span>左上肢</span><strong>{safeNumber(segmental.la ?? segmental.left_arm_kg ?? segmental.la_kg ?? segmental.segmental_muscle_la, 2)} kg</strong></div>
        <div><span>右下肢</span><strong>{safeNumber(segmental.rl ?? segmental.right_leg_kg ?? segmental.rl_kg ?? segmental.segmental_muscle_rl, 2)} kg</strong></div>
        <div><span>左下肢</span><strong>{safeNumber(segmental.ll ?? segmental.left_leg_kg ?? segmental.ll_kg ?? segmental.segmental_muscle_ll, 2)} kg</strong></div>
      </div>
    </section>
  )
}

function MemberReportDocument({ dashboard, actor, reportRef, intensity, bodyStatus, bodyLabels, muscleSnapshots, muscleSnapshotStatus }) {
  if (!dashboard?.member) return null

  const member = dashboard.member
  const today = dashboard.today_status || {}
  const assessment = dashboard.assessment || {}
  const reportDate = taipeiDateParts()
  const effectiveBodyLabels = bodyLabels || today.body_status_labels || {}
  const effectiveBodyStatus = bodyStatus || today.body_status || {}
  const selectedBodyLabels = Object.entries(effectiveBodyLabels)
    .filter(([key]) => effectiveBodyStatus?.[key])
    .map(([, label]) => label)
  const dataStatusItems = member.data_status?.items || []
  const bodyStatusText = selectedBodyLabels.length ? selectedBodyLabels.join('、') : '無'
  const goalsText = member.goals?.length ? member.goals.join('、') : '尚未設定'
  const score = assessment.page1_score || {}

  return (
    <div className="member-report-export-root" ref={reportRef} aria-hidden="true" data-report-version={REPORT_LAYOUT_VERSION}>
      <section className="member-report-page report-overview-page">
        <div className="member-report-header">
          <div>
            <div className="member-report-brand">CoachOS</div>
            <div className="member-report-company">Vitruvian Sport Technology</div>
          </div>
          <div className="member-report-title-block">
            <h1>Member Assessment Report | 會員評估報告</h1>
            <span>報告日期 {reportDate.display}</span>
          </div>
        </div>

        <section className="report-card report-profile-card">
          <div className="report-section-heading">
            <h3>Member Profile | 會員資料</h3>
            <span>{member.used_id}</span>
          </div>
          <div className="report-profile-grid">
            <div><span>姓名</span><strong>{member.user_name || '--'}</strong></div>
            <div><span>學員 ID</span><strong>{member.used_id || '--'}</strong></div>
            <div><span>年齡</span><strong>{member.age != null ? `${member.age} 歲` : '待資料收集'}</strong></div>
            <div><span>性別</span><strong>{reportGenderLabel(member.gender)}</strong></div>
            <div><span>分店</span><strong>{member.branch_id || '--'}</strong></div>
            <div><span>報告教練</span><strong>{actor?.display_name || actor?.coach_id || '--'}</strong></div>
            <div className="report-profile-wide"><span>訓練目標</span><strong>{goalsText}</strong></div>
            <div><span>今日訓練狀態</span><strong>{reportIntensityLabel(intensity || today.intensity_status)}</strong></div>
            <div><span>今日身體不適</span><strong>{bodyStatusText}</strong></div>
          </div>
        </section>

        <section className="report-card report-readiness-card">
          <div className="report-section-heading">
            <h3>資料完整度</h3>
            <span>{reportStateLabel(member.data_status?.overall_state)}</span>
          </div>
          <div className="report-readiness-grid">
            {dataStatusItems.length ? dataStatusItems.map((item) => (
              <div key={item.key}>
                <span>{item.label}</span>
                <strong>{reportStateLabel(item.state || (item.available ? 'ready' : 'missing'))}</strong>
                <small>{item.date || '—'}</small>
              </div>
            )) : <div className="report-empty">目前沒有資料完整度資訊</div>}
          </div>
        </section>

        <div className="report-score-grid">
          <section className="report-score-card">
            <span>CoachOS 體組成分數</span>
            <strong>{score.body_score ?? '--'}</strong>
            <small>{score.body_date || '--'}</small>
          </section>
          <section className="report-score-card secondary">
            <span>動康評-體測分數</span>
            <strong>{score.movement_score ?? '--'}</strong>
            <small>{score.movement_date || '--'}</small>
          </section>
        </div>

        <div className="report-trend-grid">
          <ReportTrendChart title="CoachOS 體組成分數趨勢" rows={assessment.page2_trend?.body || []} />
          <ReportTrendChart title="動康評-體測分數趨勢" rows={assessment.page2_trend?.movement || []} />
        </div>
        <ReportPageFooter page={1} />
      </section>

      <section className="member-report-page">
        <div className="member-report-section-title">
          <div><span>02</span><h2>Body Composition | 體組成分析</h2></div>
          <small>{member.user_name} · {member.used_id}</small>
        </div>
        <ReportBodyComp data={assessment.page3_bodycomp} />
        <ReportPageFooter page={2} />
      </section>

      <section className="member-report-page report-muscle-page">
        <div className="member-report-section-title">
          <div><span>03</span><h2>Risk Muscle Map | 體測待強化肌群</h2></div>
          <small>{dashboard.training?.risk_muscle_date || '最新體測'}</small>
        </div>
        <section className="report-card report-muscle-card">
          <div className="report-muscle-grid">
            <div className="report-muscle-view">
              <div className="report-muscle-label">Front | 正面</div>
              {muscleSnapshots?.front
                ? <img src={muscleSnapshots.front} alt="待強化肌群正面圖" />
                : <div className="report-muscle-placeholder">{muscleSnapshotStatus === 'error' ? '3D 肌肉圖產生失敗' : '正在產生 3D 肌肉圖...'}</div>}
            </div>
            <div className="report-muscle-view">
              <div className="report-muscle-label">Back | 背面</div>
              {muscleSnapshots?.back
                ? <img src={muscleSnapshots.back} alt="待強化肌群背面圖" />
                : <div className="report-muscle-placeholder">{muscleSnapshotStatus === 'error' ? '3D 肌肉圖產生失敗' : '正在產生 3D 肌肉圖...'}</div>}
            </div>
          </div>
        </section>
        <ReportPageFooter page={3} />
      </section>

      <section className="member-report-page">
        <div className="member-report-section-title">
          <div><span>04</span><h2>Posture Assessment | 姿態分析</h2></div>
          <small>Front 正面 · Side 側面</small>
        </div>
        <ReportMetricSection page={assessment.page4_front} title="Front | 正面姿態" />
        <ReportMetricSection page={assessment.page5_side} title="Side | 側面姿態" />
        <ReportPageFooter page={4} />
      </section>

      <section className="member-report-page">
        <div className="member-report-section-title">
          <div><span>05</span><h2>Functional Movement | 功能性動作分析</h2></div>
          <small>Bridge 橋式 · OHS 過頭深蹲 · Bird Dog 鳥狗式</small>
        </div>
        <ReportMetricSection page={assessment.page6_bridge} title="Shoulder Bridge | 橋式" />
        <ReportMetricSection page={assessment.page7_ohs} title="Overhead Squat | 過頭深蹲" />
        <ReportMetricSection page={assessment.page8_bird_dog} title="Bird Dog | 鳥狗式" />
        <ReportPageFooter page={5} />
      </section>

      <section className="member-report-page">
        <div className="member-report-section-title">
          <div><span>06</span><h2>Gait Analysis | 步態分析</h2></div>
          <small>步態評估比較</small>
        </div>
        <ReportMetricSection page={assessment.page9_gait} title="Gait Analysis | 步態分析" />
        {assessment.page9_gait?.reference_meta && (
          <section className="report-card report-reference-note">
            <h3>參考來源</h3>
            <p>{assessment.page9_gait.reference_meta.label || '參考值'}</p>
            <small>{assessment.page9_gait.reference_meta.description || ''}</small>
          </section>
        )}
        <section className="report-card report-closing-note">
          <h3>報告說明</h3>
          <p>本報告呈現 CoachOS 已收集的會員資料與評估比較結果，用於教練追蹤與運動健康促進溝通。若某項資料不足，報告會明確標示，不自行補推或產生不存在的量測值。</p>
        </section>
        <ReportPageFooter page={6} />
      </section>
    </div>
  )
}

function TrendChart({ title, rows = [], suffix = '' }) {
  if (!rows.length) {
    return (
      <div className="trend-card">
        <div className="subcard-title">{title}</div>
        <div className="empty-state compact">尚無足夠歷史資料</div>
      </div>
    )
  }

  const width = 560
  const height = 190
  const padX = 42
  const padY = 28
  const values = rows.map((r) => Number(r.value)).filter(Number.isFinite)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = rows.map((row, index) => {
    const x = rows.length === 1 ? width / 2 : padX + (index * (width - padX * 2)) / (rows.length - 1)
    const y = height - padY - ((Number(row.value) - min) / range) * (height - padY * 2)
    return { ...row, x, y }
  })

  return (
    <div className="trend-card">
      <div className="subcard-title">{title}</div>
      <svg className="trend-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line x1={padX} y1={height - padY} x2={width - padX} y2={height - padY} className="chart-axis" />
        <polyline points={points.map((p) => `${p.x},${p.y}`).join(' ')} className="chart-line" />
        {points.map((p, index) => (
          <g key={`${p.date}-${index}`}>
            <circle cx={p.x} cy={p.y} r="5.5" className="chart-dot" />
            <text x={p.x} y={p.y - 14} textAnchor="middle" className="chart-value">
              {safeNumber(p.value, 1)}{suffix}
            </text>
            <text x={p.x} y={height - 6} textAnchor="middle" className="chart-date">
              {(p.date || '').slice(5)}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

function MetricComparison({ page }) {
  if (!page?.metrics?.length || !page?.dates?.length) {
    return <div className="empty-state">尚無 {page?.title || '此項'} 最近兩次資料</div>
  }

  return (
    <div className={`comparison-layout ${page.image ? '' : 'no-figure'}`}>
      <div className="comparison-table-wrap">
        <div className="comparison-heading">
          <div>
            <div className="eyebrow">最近兩次評估</div>
            <h3>{page.title}</h3>
          </div>
          {page.reference_meta?.label && (
            <span className="reference-source" title={page.reference_meta?.description || ''}>
              {page.reference_meta.label}
            </span>
          )}
        </div>

        <div className="metric-table">
          <div className="metric-row metric-header-row" style={{ '--result-cols': page.dates.length }}>
            <div>評估指標</div>
            {page.dates.map((date, index) => <div key={`${date}-${index}`}>{date || '--'}</div>)}
          </div>
          {page.metrics.map((metric) => (
            <div className="metric-row" style={{ '--result-cols': page.dates.length }} key={metric.key}>
              <div className="metric-name-cell">
                <strong>{metric.label}</strong>
                <small>{metric.reference_label || '參考'} {metric.reference || '-'}</small>
              </div>
              {page.dates.map((_, index) => {
                const abnormal = metric.abnormal_values?.[index] === true
                const classes = [index === page.dates.length - 1 ? 'metric-current' : '', abnormal ? 'metric-abnormal' : ''].filter(Boolean).join(' ')
                return (
                  <div className={classes} key={index}>
                    {formatAssessmentValue(metric, metric.values?.[index])}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {page.image && (
        <aside className="assessment-figure">
          <div className="figure-label">指標圖例</div>
          <img src={page.image} alt={`${page.title} 指標圖例`} />
        </aside>
      )}
    </div>
  )
}

function BodyCompView({ data }) {
  if (!data) return <div className="empty-state">尚無 Body Composition 資料</div>
  const measurements = data.measurements || {}
  const derived = data.derived_metrics || {}
  const segmental = data.segmental_muscle || {}
  const coachos = data.scores?.coachos
  const comparisons = data.comparisons || {}

  const hiddenMeasurementKeys = new Set(['body_fat_reference_low_pct', 'body_fat_reference_high_pct', 'fat_free_mass_kg'])
  const renderTile = (scope, key, value, accent = false) => {
    const comparison = comparisons[`${scope}.${key}`]
    const abnormal = comparison?.abnormal === true
    return (
      <div className={`data-tile ${accent ? 'accent' : ''} ${abnormal ? 'data-abnormal' : ''}`} key={`${scope}.${key}`}>
        <span>{scope === 'measurements' ? (BODYCOMP_LABELS[key] || key) : (DERIVED_LABELS[key] || key)}</span>
        <strong>{scope === 'measurements' ? formatBodyValue(key, value) : (key.includes('pct') ? `${safeNumber(value, 1)}%` : safeNumber(value, 1))}</strong>
        {comparison?.reference && <small>參考 {comparison.reference}</small>}
      </div>
    )
  }

  return (
    <div className="bodycomp-layout">
      <div className="body-score-panel">
        <div className="eyebrow">體組成分析</div>
        <div className="score-orbit">
          <span>{coachos?.value ?? '--'}</span>
          <small>BCS</small>
        </div>
        <div className="score-caption">{data.date || '--'} · {(data.device_type || '').toUpperCase()}</div>
      </div>

      <div className="body-data-grid">
        {Object.entries(measurements).filter(([key]) => !hiddenMeasurementKeys.has(key)).map(([key, value]) => renderTile('measurements', key, value, false))}
        {Object.entries(derived).map(([key, value]) => renderTile('derived_metrics', key, value, true))}
      </div>

      <div className="segment-card">
        <div className="subcard-title">節段肌肉量</div>
        <div className="segment-grid">
          <div><span>右上肢</span><strong>{safeNumber(segmental.ra ?? segmental.right_arm_kg ?? segmental.ra_kg ?? segmental.segmental_muscle_ra, 2)} kg</strong></div>
          <div><span>左上肢</span><strong>{safeNumber(segmental.la ?? segmental.left_arm_kg ?? segmental.la_kg ?? segmental.segmental_muscle_la, 2)} kg</strong></div>
          <div><span>右下肢</span><strong>{safeNumber(segmental.rl ?? segmental.right_leg_kg ?? segmental.rl_kg ?? segmental.segmental_muscle_rl, 2)} kg</strong></div>
          <div><span>左下肢</span><strong>{safeNumber(segmental.ll ?? segmental.left_leg_kg ?? segmental.ll_kg ?? segmental.segmental_muscle_ll, 2)} kg</strong></div>
        </div>
      </div>
    </div>
  )
}

function AssessmentPanel({ assessment, activeTab }) {
  if (!assessment) return <div className="empty-state">讀取評估資料中...</div>

  if (activeTab === 'score') {
    const p = assessment.page1_score || {}
    return (
      <div className="score-page">
        <div className="score-card primary-score">
          <span>CoachOS 體組成分數</span>
          <strong>{p.body_score ?? '--'}</strong>
          <small>{p.body_date || '--'}</small>
        </div>
        <div className="score-card secondary-score">
          <span>動康評-體測分數</span>
          <strong>{p.movement_score ?? '--'}</strong>
          <small>{p.movement_date || '--'}</small>
        </div>

      </div>
    )
  }

  if (activeTab === 'trend') {
    return (
      <div className="trend-grid">
        <TrendChart title="CoachOS 體組成分數 最近五次" rows={assessment.page2_trend?.body || []} />
        <TrendChart title="動康評-體測分數 最近五次" rows={assessment.page2_trend?.movement || []} />
      </div>
    )
  }

  if (activeTab === 'bodycomp') return <BodyCompView data={assessment.page3_bodycomp} />
  if (activeTab === 'front') return <MetricComparison page={assessment.page4_front} />
  if (activeTab === 'side') return <MetricComparison page={assessment.page5_side} />
  if (activeTab === 'bridge') return <MetricComparison page={assessment.page6_bridge} />
  if (activeTab === 'ohs') return <MetricComparison page={assessment.page7_ohs} />
  if (activeTab === 'bird') return <MetricComparison page={assessment.page8_bird_dog} />
  if (activeTab === 'gait') return <MetricComparison page={assessment.page9_gait} />
  return null
}

function TrainingModal({ open, onClose, data }) {
  if (!open || !data) return null
  const weekly = data.weekly_structure || {}
  const phases = [
    ['熱身', 'warmup', data.warmup],
    ['主訓練', 'main', data.main],
    ['緩和', 'cooldown', data.cooldown],
  ]

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="plan-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">FULL TRAINING PLAN</div>
            <h2>訓練計畫</h2>
          </div>
          <button className="icon-button" onClick={onClose}>×</button>
        </div>

        <div className="weekly-strip">
          <div><span>每週</span><strong>{weekly.days_per_week ?? '--'} 天</strong></div>
          <div><span>單次</span><strong>{weekly.session_duration_min ?? '--'} 分鐘</strong></div>
          <div><span>強度</span><strong>{trainingIntensityLabel(weekly.intensity_level)}</strong></div>
        </div>

        {data.progression && (
          <div className="plan-note highlight-note">
            <span>進程</span>
            <p>{data.progression}</p>
          </div>
        )}

        <Muscle3DViewer riskMuscles={data.risk_muscles || []} sourceDate={data.risk_muscle_date || ''} />

        {phases.map(([title, phaseKey, phase]) => (
          <section className="plan-section" key={title}>
            <div className="plan-section-head">
              <h3>{title}</h3>
              <span>{phase?.duration_min ?? '--'} min</span>
            </div>
            <p className="plan-note-text">{phase?.note || '--'}</p>
            <div className="exercise-cloud">
              {(phase?.exercises || []).map((x) => <span key={x}>{x}</span>)}
            </div>
            {!phase?.exercises?.length && (
              <div className="phase-empty-notice">
                {phase?.empty_message || EMPTY_EXERCISE_MESSAGE}
              </div>
            )}

            {phaseKey === 'main' && !!phase?.issue_exercises?.length && (
              <div className="issue-table">
                <div className="issue-row issue-header"><div>加強建議</div><div>體測問題點</div><div>原因</div></div>
                {phase.issue_exercises.map((row, index) => (
                  <div className="issue-row" key={`${row.issue_problem || row.issue_key}-${row.pose}-${index}`}>
                    <div>{row.pose}</div>
                    <div className="issue-problem">{getIssueDisplayLabel(row)}</div>
                    <div>{row.reason || '--'}</div>
                  </div>
                ))}
              </div>
            )}
          </section>
        ))}
      </div>
    </div>
  )
}

function LoginPanel({ onLogin, status, busy }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const submit = (e) => {
    e.preventDefault()
    onLogin(email, password)
  }

  return (
    <div className="login-shell">
      <div className="login-art login-art-a" />
      <div className="login-art login-art-b" />
      <form className="login-card" onSubmit={submit}>
        <div className="brand-mark large"><span>V</span></div>
        <div className="eyebrow">VITRUVIAN SPORT TECHNOLOGY</div>
        <h1>CoachOS</h1>
        <p>教練決策與會員追蹤工作台</p>
        <label>
          Email
          <input value={email} onChange={(e) => setEmail(e.target.value)} type="email" autoComplete="email" required />
        </label>
        <label>
          Password
          <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" autoComplete="current-password" required />
        </label>
        <button className="primary-button full" type="submit" disabled={busy}>{busy ? '登入中...' : '登入 CoachOS'}</button>
        {status && <div className="login-status">{status}</div>}
      </form>
    </div>
  )
}

export default function CoachDashboard() {
  const [session, setSession] = useState(loadStoredSession)
  const [actor, setActor] = useState(null)
  const [loginStatus, setLoginStatus] = useState('')
  const [loginBusy, setLoginBusy] = useState(false)
  const [members, setMembers] = useState([])
  const [selectedKey, setSelectedKey] = useState('')
  const [dashboard, setDashboard] = useState(null)
  const [loading, setLoading] = useState(false)
  const [generateStatus, setGenerateStatus] = useState('')
  const [generating, setGenerating] = useState(false)
  const [intensity, setIntensity] = useState('moderate')
  const [bodyStatus, setBodyStatus] = useState({})
  const [activeAssessment, setActiveAssessment] = useState('score')
  const [planOpen, setPlanOpen] = useState(false)
  const [memberMenuOpen, setMemberMenuOpen] = useState(false)
  const [reportExporting, setReportExporting] = useState(false)
  const [reportMuscleSnapshots, setReportMuscleSnapshots] = useState(null)
  const [reportMuscleSnapshotStatus, setReportMuscleSnapshotStatus] = useState('idle')
  const reportRef = useRef(null)

  const selectedMember = useMemo(
    () => members.find((m) => `${m.branch_id}::${m.used_id}` === selectedKey),
    [members, selectedKey],
  )

  const updateSession = (next) => {
    saveSession(next)
    setSession(next)
  }

  const forceLogout = () => {
    saveSession(null)
    setSession(null)
    setActor(null)
    setMembers([])
    setSelectedKey('')
    setDashboard(null)
  }

  const handleApiError = (error, fallback) => {
    if (error.response?.status === 401) {
      forceLogout()
      return '登入已失效，請重新登入'
    }
    return error.response?.data?.detail || error.message || fallback
  }

  const loadMe = async (activeSession = session) => {
    const response = await coachRequest(activeSession, updateSession, { method: 'get', url: '/api/coach/me' })
    setActor(response.data.user)
    return response.data.user
  }

  const login = async (email, password) => {
    setLoginBusy(true)
    setLoginStatus('驗證中...')
    try {
      const nextSession = await firebaseSignIn(email, password)
      updateSession(nextSession)
      await loadMe(nextSession)
      setLoginStatus('')
    } catch (error) {
      console.error(error)
      saveSession(null)
      setSession(null)
      setActor(null)
      setLoginStatus(error.response?.data?.error?.message || error.response?.data?.detail || error.message || '登入失敗')
    } finally {
      setLoginBusy(false)
    }
  }

  useEffect(() => {
    if (!session || actor) return
    loadMe(session).catch((error) => {
      console.error(error)
      setLoginStatus(handleApiError(error, '無法讀取 Coach 權限'))
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, actor])

  useEffect(() => {
    if (!session || !actor) return
    const loadMembers = async () => {
      try {
        const response = await coachRequest(session, updateSession, { method: 'get', url: '/api/coach/members' })
        const rows = response.data.members || []
        setMembers(rows)
        setSelectedKey((prev) => {
          if (prev && rows.some((m) => `${m.branch_id}::${m.used_id}` === prev)) return prev
          return rows.length ? `${rows[0].branch_id}::${rows[0].used_id}` : ''
        })
      } catch (error) {
        console.error(error)
        setGenerateStatus(`會員名單讀取失敗：${handleApiError(error, '未知錯誤')}`)
      }
    }
    loadMembers()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, actor])

  useEffect(() => {
    if (!session || !actor || !selectedMember) return
    loadDashboard(selectedMember)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedKey])

  const loadDashboard = async (member) => {
    setLoading(true)
    setGenerateStatus('')
    try {
      const response = await coachRequest(session, updateSession, {
        method: 'get',
        url: `/api/coach/member/${encodeURIComponent(member.used_id)}`,
        params: { branch_id: member.branch_id, include_communication: true },
      })
      const data = response.data.data
      setDashboard(data)
      setIntensity(data.today_status?.intensity_status || 'moderate')
      setBodyStatus(data.today_status?.body_status || {})
    } catch (error) {
      console.error(error)
      setDashboard(null)
      setGenerateStatus(handleApiError(error, '會員資料讀取失敗'))
    } finally {
      setLoading(false)
    }
  }

  const toggleBody = (key) => setBodyStatus((prev) => ({ ...prev, [key]: !prev[key] }))

  const generate = async () => {
    if (!dashboard || !selectedMember || !session) return
    setGenerating(true)
    setGenerateStatus('正在更新 Today Status、Rule Engine 與溝通建議...')
    try {
      const response = await coachRequest(session, updateSession, {
        method: 'post',
        url: '/api/coach/generate',
        data: {
          branch_id: selectedMember.branch_id,
          used_id: dashboard.member.used_id,
          user_name: dashboard.member.user_name,
          intensity_status: intensity,
          body_status: bodyStatus,
        },
      })
      const next = response.data.data
      setDashboard(next)
      setIntensity(next.today_status?.intensity_status || intensity)
      setBodyStatus(next.today_status?.body_status || bodyStatus)
      setGenerateStatus(`已更新今日建議 · ${next.communication?.communication_style_label || '新版本'}`)
    } catch (error) {
      console.error(error)
      setGenerateStatus(`生成失敗：${handleApiError(error, '未知錯誤')}`)
    } finally {
      setGenerating(false)
    }
  }

  const exportMemberReport = async () => {
    if (!dashboard?.member || !reportRef.current || reportExporting) return
    setReportExporting(true)
    console.info(`[CoachOS] member report layout ${REPORT_LAYOUT_VERSION}`)

    try {
      setReportMuscleSnapshotStatus('loading')
      try {
        const muscleSnapshots = await renderMuscleReportSnapshots(dashboard.training?.risk_muscles || [])
        setReportMuscleSnapshots(muscleSnapshots)
        setReportMuscleSnapshotStatus('ready')
      } catch (muscleError) {
        console.error('3D 肌肉圖報告快照失敗:', muscleError)
        setReportMuscleSnapshots(null)
        setReportMuscleSnapshotStatus('error')
      }

      if (document.fonts?.ready) await document.fonts.ready
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))

      const reportImages = Array.from(reportRef.current.querySelectorAll('.report-muscle-view img'))
      await Promise.all(reportImages.map((image) => {
        if (image.complete && image.naturalWidth > 0) return Promise.resolve()
        if (typeof image.decode === 'function') return image.decode().catch(() => undefined)
        return new Promise((resolve) => {
          image.onload = resolve
          image.onerror = resolve
        })
      }))

      const pages = Array.from(reportRef.current.querySelectorAll('.member-report-page'))
      if (!pages.length) throw new Error('找不到報告頁面')

      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
        compress: true,
      })

      for (let index = 0; index < pages.length; index += 1) {
        const pageElement = pages[index]
        const canvas = await html2canvas(pageElement, {
          scale: 2,
          backgroundColor: '#ffffff',
          useCORS: true,
          logging: false,
          width: pageElement.offsetWidth,
          height: pageElement.offsetHeight,
          windowWidth: pageElement.offsetWidth,
          windowHeight: pageElement.offsetHeight,
          scrollX: 0,
          scrollY: 0,
          removeContainer: true,
        })
        // PNG avoids JPEG text/line artifacts in tables and charts.
        const imageData = canvas.toDataURL('image/png')
        if (index > 0) pdf.addPage('a4', 'portrait')
        pdf.addImage(imageData, 'PNG', 0, 0, 210, 297, undefined, 'FAST')
      }

      const datePart = taipeiDateParts().compact
      const memberId = sanitizeReportFilenamePart(dashboard.member.used_id)
      pdf.save(`${memberId}_${datePart}.pdf`)
    } catch (error) {
      console.error('會員報告輸出失敗:', error)
      setReportMuscleSnapshotStatus('error')
      window.alert(`會員報告輸出失敗：${error.message || '未知錯誤'}`)
    } finally {
      setReportExporting(false)
    }
  }

  if (!session || !actor) return <LoginPanel onLogin={login} status={loginStatus} busy={loginBusy} />

  const member = dashboard?.member
  const training = dashboard?.training
  const communication = dashboard?.communication
  const assessment = dashboard?.assessment
  const bodyLabels = dashboard?.today_status?.body_status_labels || {}

  return (
    <div className="coach-app">
      <header className="topbar">
        <div className="brand-zone">
          <div className="brand-mark"><span>V</span></div>
          <div>
            <div className="brand-title">Coach<span>OS</span></div>
            <div className="brand-subtitle">Vitruvian Sport Technology</div>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="role-chip"><span>{roleLabel(actor.role)}</span>{actor.display_name || actor.coach_id}</div>
          <div className="member-switcher">
            <button className="ghost-button" onClick={() => setMemberMenuOpen((v) => !v)}>
              {selectedMember ? `${selectedMember.user_name} · ${selectedMember.used_id}` : '切換會員'} <span>⌄</span>
            </button>
            {memberMenuOpen && (
              <div className="member-menu">
                {members.map((m) => {
                  const key = `${m.branch_id}::${m.used_id}`
                  return (
                    <button
                      key={key}
                      className={key === selectedKey ? 'member-menu-item active' : 'member-menu-item'}
                      onClick={() => { setSelectedKey(key); setMemberMenuOpen(false) }}
                    >
                      <strong>{m.user_name}</strong>
                      <span>{m.branch_id} · {m.used_id}</span>
                    </button>
                  )
                })}
                {!members.length && <div className="member-menu-empty">目前沒有可見會員</div>}
              </div>
            )}
          </div>
          <button
            className="report-button"
            onClick={exportMemberReport}
            disabled={!dashboard?.member || loading || reportExporting}
          >
            {reportExporting ? '報告產生中...' : '學員報告輸出'}
          </button>
          <button className="text-button" onClick={forceLogout}>登出</button>
        </div>
      </header>

      <main className="dashboard-shell">
        <section className="page-intro">
          <div>
            <div className="eyebrow">SMART COACHING WORKSPACE</div>
            <h1>教練工作台</h1>
            {actor.role !== 'coach' && <p>{actor.role === 'owner' ? '全公司會員視角' : `${actor.branch_id} 分店管理視角`}</p>}
          </div>
          <div className="system-pill"><span /> SYSTEM ONLINE</div>
        </section>

        {loading ? (
          <div className="loading-panel"><div className="spinner" />正在整理會員最新資料...</div>
        ) : !dashboard ? (
          <div className="empty-state large">{generateStatus || '沒有可顯示的會員資料'}</div>
        ) : (
          <>
            <section className="hero-grid">
              <aside className="surface-card member-card">
                <div className="card-heading-row">
                  <div className="eyebrow">MEMBER PROFILE</div>
                  <span className="tiny-badge">{member.used_id}</span>
                </div>

                <div className="member-profile-main">
                  <div className="avatar-ring"><div>{initials(member.user_name)}</div></div>
                  <div>
                    <h2>{member.user_name}</h2>
                    <p>{member.age != null ? `${member.age} 歲` : '年齡待資料收集'} · {member.gender === 'female' ? '女性' : member.gender === 'male' ? '男性' : '性別待資料收集'} · {member.branch_id}</p>
                  </div>
                </div>

                <div className="goal-block">
                  <span>目標</span>
                  <strong>{member.goals?.length ? member.goals.join(' · ') : '尚未設定'}</strong>
                </div>

                <div className="section-label-row data-readiness-label">
                  <span>資料完整度</span>
                  <small>
                    {member.data_status?.overall_state === 'ready'
                      ? '資料齊全'
                      : member.data_status?.overall_state === 'missing'
                        ? '待資料收集'
                        : '部分資料待補'}
                  </small>
                </div>
                <div className="data-status-grid">
                  {(member.data_status?.items || []).map((item) => {
                    const state = item.state || (item.available ? 'ready' : 'missing')
                    const primaryText =
                      state === 'ready'
                        ? (item.date || '已有資料')
                        : state === 'partial'
                          ? `${item.date || '已有資料'} · 部分資料`
                          : '待資料收集'

                    return (
                      <div className={`data-status-item ${state}`} key={item.key}>
                        <span className="status-light" aria-hidden="true" />
                        <div>
                          <strong>{item.label}</strong>
                          <small>{primaryText}</small>
                          {state === 'partial' && item.missing_labels?.length > 0 && (
                            <small className="status-detail">缺：{item.missing_labels.join('、')}</small>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>

                <div className="section-label-row"><span>今日狀態</span><small>未設定時預設正常</small></div>
                <div className="segmented-control">
                  {INTENSITY_OPTIONS.map((opt) => (
                    <button key={opt.key} className={intensity === opt.key ? 'selected' : ''} onClick={() => setIntensity(opt.key)}>
                      <strong>{opt.label}</strong><small>{opt.caption}</small>
                    </button>
                  ))}
                </div>

                <div className="section-label-row"><span>今日是否有身體部位不適</span><small>可複選</small></div>
                <div className="body-chip-grid">
                  {Object.entries(bodyLabels).map(([key, label]) => (
                    <button key={key} className={bodyStatus[key] ? 'body-chip selected' : 'body-chip'} onClick={() => toggleBody(key)}>
                      <span className="check-dot">{bodyStatus[key] ? '●' : '○'}</span>{label}
                    </button>
                  ))}
                </div>

                <button className="primary-button full generate-button" onClick={generate} disabled={generating}>
                  {generating ? '生成中...' : '✦ 生成今日建議'}
                </button>
                {generateStatus && <div className="generate-status">{generateStatus}</div>}
              </aside>

              <div className="right-stack">
                <section className="surface-card ai-card">
                  <div className="ai-card-header">
                    <div className="ai-icon">✦</div>
                    <div>
                      <div className="eyebrow">AI COACHING PLAN</div>
                      <h2>AI 教練建議</h2>
                    </div>
                  </div>

                  {training?.available === false && (
                    <div className="training-unavailable">
                      <strong>訓練計畫待資料</strong>
                      <span>{training.unavailable_reason || '待體測或體組成資料完成後產生訓練計畫'}</span>
                    </div>
                  )}

                  <div className="phase-grid">
                    {[
                      ['01', '熱身', training?.warmup],
                      ['02', '主訓練', training?.main],
                      ['03', '緩和', training?.cooldown],
                    ].map(([num, title, phase]) => (
                      <div className="phase-card" key={title}>
                        <div className="phase-top"><span>{num}</span><strong>{title}</strong><em>{phase?.duration_min ?? '--'} min</em></div>
                        <p>{phase?.note || '--'}</p>
                        <div className="mini-exercises">
                          {(phase?.exercises || []).slice(0, 4).map((x) => <span key={x}>{x}</span>)}
                          {(phase?.exercises || []).length > 4 && <span>+{phase.exercises.length - 4}</span>}
                          {!phase?.exercises?.length && (
                            <span className="phase-empty-chip">
                              {phase?.empty_message || EMPTY_EXERCISE_MESSAGE}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="milestone-box">
                    <div className="milestone-title"><span>階段性目標</span><small>追蹤節點</small></div>
                    <div className="milestone-list">
                      {(training?.milestones || []).map((m, index) => (
                        <div className="milestone-item" key={index}>
                          <div className="milestone-dot">{m.week ? `W${m.week}` : '•'}</div>
                          <div>{m.target}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <button className="outline-button plan-button" onClick={() => setPlanOpen(true)} disabled={training?.available === false}>查看訓練計畫 <span>→</span></button>
                </section>

                <section className="surface-card communication-card">
                  <div className="communication-head">
                    <div>
                      <div className="eyebrow">COACH COMMUNICATION</div>
                      <h2>溝通建議</h2>
                    </div>
                    <div className="communication-chips">
                      <span className="style-chip">{communication?.communication_style_label || '自動風格'}</span>
                      {communication?.interaction_type_label && (
                        <span className="interaction-chip">互動｜{communication.interaction_type_label}</span>
                      )}
                    </div>
                  </div>
                  <p className="advice-text">{communication?.communication_advice || '--'}</p>
                  <div className="coach-script"><span>教練可以這樣說</span>{communication?.coach_script || '--'}</div>
                  <div className="tone-row">
                    <span>語氣</span>
                    {(communication?.tone_tags || []).map((tag) => <b key={tag}>{tag}</b>)}
                  </div>
                  <p className="tone-guidance">{communication?.tone_guidance}</p>
                </section>
              </div>
            </section>

            <section className="surface-card assessment-card">
              <div className="assessment-title-row">
                <div>
                  <div className="eyebrow">評估歷史</div>
                  <h2>評估比較</h2>
                </div>
                <div className="micro-copy">最近兩次評估比較</div>
              </div>

              <div className="assessment-tabs">
                {ASSESSMENT_TABS.map(([key, label]) => (
                  <button key={key} className={activeAssessment === key ? 'active' : ''} onClick={() => setActiveAssessment(key)}>{label}</button>
                ))}
              </div>

              <div className="assessment-content">
                <AssessmentPanel assessment={assessment} activeTab={activeAssessment} />
              </div>
            </section>
          </>
        )}
      </main>

      <TrainingModal open={planOpen} onClose={() => setPlanOpen(false)} data={training} />
      <MemberReportDocument
        dashboard={dashboard}
        actor={actor}
        reportRef={reportRef}
        intensity={intensity}
        bodyStatus={bodyStatus}
        bodyLabels={bodyLabels}
        muscleSnapshots={reportMuscleSnapshots}
        muscleSnapshotStatus={reportMuscleSnapshotStatus}
      />
    </div>
  )
}
