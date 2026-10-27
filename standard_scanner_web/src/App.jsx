import { useState, useEffect } from 'react'
import imageCompression from 'browser-image-compression'
import axios from 'axios'
import {
  clearSharedSession,
  ensureFreshSharedSession,
  loadSharedSession,
  redirectToPortal,
  saveSharedSession,
} from './sharedAuth'
import './App.css'

const FIREBASE_API_KEY = import.meta.env.VITE_FIREBASE_API_KEY || ''
const CLOUD_API_URL = import.meta.env.VITE_CLOUD_API_URL || 'http://127.0.0.1:8000'
const roleCapabilities = (role, rawRoles) => {
  const cleanRole = String(role || '').trim().toLowerCase()
  if (Array.isArray(rawRoles) && rawRoles.length > 0) {
    return [...new Set(
      rawRoles
        .map((item) => String(item || '').trim().toLowerCase())
        .filter((item) => ['coach', 'manager', 'owner'].includes(item))
    )]
  }
  if (cleanRole === 'manager') return ['manager', 'coach']
  if (cleanRole === 'owner') return ['owner', 'coach']
  if (cleanRole === 'coach') return ['coach']
  return []
}

const QUESTIONNAIRE_PAGE_NUMBERS = [1, 2, 3, 4]

const createEmptyQuestionnaireSlots = () =>
  Object.fromEntries(
    QUESTIONNAIRE_PAGE_NUMBERS.map((pageNumber) => [
      pageNumber,
      { file: null, preview: null },
    ])
  )

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(Boolean(loadSharedSession()))
  const [authInfo, setAuthInfo] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)
  const [authError, setAuthError] = useState('')

  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState('')
  const [branches, setBranches] = useState([])
  const [selectedBranch, setSelectedBranch] = useState('')

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('等待操作')
  const [lastResult, setLastResult] = useState(null)

  // 掃描模式：body = 身體組成；questionnaire = CoachOS 標準健康/生活問卷
  const [scanMode, setScanMode] = useState('body')

  // 問卷改為固定 Page 1～Page 4 slot。
  // 同一頁重拍時直接 replace，不再把照片追加成新的陣列項目。
  const [questionnaireSlots, setQuestionnaireSlots] = useState(
    createEmptyQuestionnaireSlots
  )
  // 第一張問卷照片拍下時鎖定本次掃描的學員。
  // 若教練切換學員，尚未送出的 slot 會全部清空並解除鎖定。
  const [questionnaireLockedUserId, setQuestionnaireLockedUserId] = useState('')
  const [questionnaireStatus, setQuestionnaireStatus] = useState('等待操作')
  const [questionnaireResult, setQuestionnaireResult] = useState(null)

  const questionnairePageNumbers = QUESTIONNAIRE_PAGE_NUMBERS.filter(
    (pageNumber) => questionnaireSlots[pageNumber]?.file
  )
  const questionnairePageCount = questionnairePageNumbers.length

  const getIdToken = async () => {
    try {
      const session = loadSharedSession()
      const fresh = await ensureFreshSharedSession(session, FIREBASE_API_KEY)
      saveSharedSession(fresh)
      return fresh.idToken
    } catch (error) {
      clearSharedSession()
      redirectToPortal('/scanner/')
      throw error
    }
  }

  const bootstrapSharedLogin = async () => {
    const session = loadSharedSession()
    if (!session?.idToken) {
      redirectToPortal('/scanner/')
      return
    }

    setAuthError('')
    setAuthLoading(true)

    try {
      const fresh = await ensureFreshSharedSession(session, FIREBASE_API_KEY)
      saveSharedSession(fresh)

      // Backend remains the authority for company / branch / coach / role.
      const meResponse = await axios.get(`${CLOUD_API_URL}/api/me`, {
        headers: { Authorization: `Bearer ${fresh.idToken}` },
      })
      const actor = meResponse.data.actor || {}
      const roles = roleCapabilities(actor.role, actor.roles)

      if (!roles.includes('coach')) {
        const permissionError = new Error('此帳號沒有 coach 上傳權限')
        permissionError.isPermissionError = true
        throw permissionError
      }

      const newAuthInfo = {
        company_id: actor.company_id,
        branch_id: actor.branch_id || null,
        coach_id: actor.coach_id,
        role: actor.role,
        roles,
      }

      setAuthInfo(newAuthInfo)
      setIsLoggedIn(true)
      setSelectedBranch(newAuthInfo.role === 'owner' ? '' : newAuthInfo.branch_id || '')
    } catch (error) {
      console.error('Shared login bootstrap failed:', error)
      const status = error.response?.status

      // Only an actual authentication failure invalidates the shared portal session.
      // Network/CORS/backend errors must NOT erase a valid portal login.
      if (status === 401) {
        clearSharedSession()
        redirectToPortal('/scanner/')
        return
      }

      const detail = error.response?.data?.detail || error.message || '未知錯誤'
      const likelyCors = !error.response && /network error/i.test(String(error.message || ''))
      setAuthError(
        likelyCors
          ? `Scanner API 無法連線。共用登入仍保留；請確認 Scanner API CORS 已允許目前入口網址。(${detail})`
          : `Scanner 登入驗證失敗：${detail}`
      )
      setIsLoggedIn(true)
    } finally {
      setAuthLoading(false)
    }
  }

  useEffect(() => {
    bootstrapSharedLogin()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const fetchBranches = async () => {
    try {
      const token = await getIdToken()
      const response = await axios.get(`${CLOUD_API_URL}/api/branches`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      setBranches(response.data.branches || [])
    } catch (error) {
      console.error('無法抓取分店名單:', error)
      setBranches([])
      setUploadStatus(
        `⚠️ 分店名單獲取失敗：${error.response?.data?.detail || error.message || '請重新登入'}`
      )
    }
  }

  const fetchUsers = async (branchId) => {
    if (!authInfo || !branchId) {
      setUsers([])
      setSelectedUser('')
      return
    }

    try {
      const token = await getIdToken()
      const response = await axios.get(`${CLOUD_API_URL}/api/users`, {
        params: {
          company_id: authInfo.company_id,
          branch_id: branchId,
          coach_id: authInfo.coach_id,
        },
        headers: { Authorization: `Bearer ${token}` },
      })

      const resultUsers = response.data.users || []
      setUsers(resultUsers)
      setSelectedUser((current) =>
        resultUsers.some((user) => user.id === current)
          ? current
          : resultUsers[0]?.id || ''
      )
    } catch (error) {
      console.error('無法抓取雲端名單:', error)
      setUsers([])
      setSelectedUser('')
      setUploadStatus(
        `⚠️ 雲端名單獲取失敗：${error.response?.data?.detail || error.message || '請確認 API 狀態'}`
      )
    }
  }

  useEffect(() => {
    if (!isLoggedIn || !authInfo) return

    if (authInfo.role === 'owner') {
      setSelectedBranch('')
      setUsers([])
      setSelectedUser('')
      fetchBranches()
    } else {
      setBranches([])
      setSelectedBranch(authInfo.branch_id || '')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    isLoggedIn,
    authInfo?.role,
    authInfo?.company_id,
    authInfo?.branch_id,
    authInfo?.coach_id,
  ])

  useEffect(() => {
    if (isLoggedIn && authInfo && selectedBranch) {
      fetchUsers(selectedBranch)
    } else if (isLoggedIn && authInfo?.role === 'owner' && !selectedBranch) {
      setUsers([])
      setSelectedUser('')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, authInfo?.company_id, authInfo?.coach_id, selectedBranch])


  const handleLogout = () => {
    clearSharedSession()

    setIsLoggedIn(false)
    setAuthInfo(null)
    setBranches([])
    setSelectedBranch('')
    setUsers([])
    setSelectedUser('')
    setFile(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setLastResult(null)
    setUploadStatus('等待操作')
    setScanMode('body')
    Object.values(questionnaireSlots).forEach((slot) => {
      if (slot?.preview) URL.revokeObjectURL(slot.preview)
    })
    setQuestionnaireSlots(createEmptyQuestionnaireSlots())
    setQuestionnaireLockedUserId('')
    setQuestionnaireResult(null)
    setQuestionnaireStatus('等待操作')
    window.location.assign('/')
  }

  const handleImageUpload = async (event) => {
    const imageFile = event.target.files?.[0]
    if (!imageFile) return

    setUploadStatus('🔄 圖片壓縮中...')

    try {
      const compressedFile = await imageCompression(imageFile, {
        maxSizeMB: 0.5,
        maxWidthOrHeight: 1024,
        useWebWorker: true,
      })

      setFile(compressedFile)
      setPreview(URL.createObjectURL(compressedFile))
      setUploadStatus(
        `✅ 壓縮完成！(${(compressedFile.size / 1024).toFixed(1)}KB)`
      )
    } catch (error) {
      console.error('圖片壓縮失敗:', error)
      setUploadStatus('❌ 圖片壓縮失敗')
    }
  }


  const revokeQuestionnaireSlotPreviews = (slots) => {
    Object.values(slots || {}).forEach((slot) => {
      if (slot?.preview) URL.revokeObjectURL(slot.preview)
    })
  }

  const clearQuestionnaireDraft = (statusMessage = '等待操作') => {
    revokeQuestionnaireSlotPreviews(questionnaireSlots)
    setQuestionnaireSlots(createEmptyQuestionnaireSlots())
    setQuestionnaireLockedUserId('')
    setQuestionnaireStatus(statusMessage)
  }

  const handleBranchChange = (event) => {
    const nextBranch = event.target.value
    if (nextBranch === selectedBranch) return

    const hadPendingQuestionnaire = questionnairePageCount > 0
    const hadPendingBodyPhoto = Boolean(file || preview)

    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setUploadStatus(
      hadPendingBodyPhoto
        ? '⚠️ 已切換分店，未送出的身體組成照片已自動清空'
        : '等待操作'
    )

    revokeQuestionnaireSlotPreviews(questionnaireSlots)
    setQuestionnaireSlots(createEmptyQuestionnaireSlots())
    setQuestionnaireLockedUserId('')
    setQuestionnaireStatus(
      hadPendingQuestionnaire
        ? '⚠️ 已切換分店，未送出的問卷照片已自動清空'
        : '等待操作'
    )

    setLastResult(null)
    setQuestionnaireResult(null)
    setUsers([])
    setSelectedUser('')
    setSelectedBranch(nextBranch)
  }

  const handleSelectedUserChange = (event) => {
    const nextUserId = event.target.value
    if (nextUserId === selectedUser) return

    const hadPendingQuestionnaire = questionnairePageCount > 0
    const hadPendingBodyPhoto = Boolean(file || preview)

    // 切換學員時，所有尚未送出的照片都清空，避免 A 的照片被送到 B。
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setUploadStatus(
      hadPendingBodyPhoto ? '⚠️ 已切換學員，未送出的身體組成照片已自動清空' : '等待操作'
    )

    revokeQuestionnaireSlotPreviews(questionnaireSlots)
    setQuestionnaireSlots(createEmptyQuestionnaireSlots())
    setQuestionnaireLockedUserId('')
    setQuestionnaireStatus(
      hadPendingQuestionnaire
        ? '⚠️ 已切換學員，未送出的問卷照片已自動清空'
        : '等待操作'
    )

    // 避免畫面繼續顯示上一位學員的解析結果。
    setLastResult(null)
    setQuestionnaireResult(null)
    setSelectedUser(nextUserId)
  }

  const handleQuestionnaireImageUpload = async (pageNumber, event) => {
    const imageFile = event.target.files?.[0]
    if (!imageFile) return

    if (!selectedUser) {
      alert('請先選擇學員！')
      event.target.value = ''
      return
    }

    // 正常情況下 selectedUser 與 locked user 永遠相同；
    // 若狀態異常，拒絕混入不同學員的照片。
    if (
      questionnaireLockedUserId &&
      questionnaireLockedUserId !== selectedUser
    ) {
      alert('本次問卷已鎖定其他學員，請先切換學員以清空未送出的照片。')
      event.target.value = ''
      return
    }

    const replacing = Boolean(questionnaireSlots[pageNumber]?.file)
    setQuestionnaireStatus(
      `🔄 正在${replacing ? '重拍並取代' : '加入'} Page ${pageNumber}...`
    )

    try {
      // 問卷文字較細，保留比身體組成報表更高解析度。
      const compressedFile = await imageCompression(imageFile, {
        maxSizeMB: 1.2,
        maxWidthOrHeight: 1800,
        useWebWorker: true,
      })

      const previewUrl = URL.createObjectURL(compressedFile)

      setQuestionnaireSlots((prev) => {
        const oldPreview = prev[pageNumber]?.preview
        if (oldPreview) URL.revokeObjectURL(oldPreview)

        return {
          ...prev,
          [pageNumber]: {
            file: compressedFile,
            preview: previewUrl,
          },
        }
      })

      // 第一張問卷照片拍下時，鎖定本次 session 的學員。
      if (!questionnaireLockedUserId) {
        setQuestionnaireLockedUserId(selectedUser)
      }

      setQuestionnaireStatus(
        `✅ Page ${pageNumber} ${replacing ? '已重拍取代' : '已加入'}；本次問卷已鎖定目前學員`
      )

      // 允許手機再次選到同一檔名，方便同頁重拍。
      event.target.value = ''
    } catch (error) {
      console.error('問卷圖片壓縮失敗:', error)
      setQuestionnaireStatus(`❌ Page ${pageNumber} 圖片壓縮失敗`)
      event.target.value = ''
    }
  }

  const removeQuestionnairePage = (pageNumber) => {
    const targetPreview = questionnaireSlots[pageNumber]?.preview
    if (targetPreview) URL.revokeObjectURL(targetPreview)

    const remainingCount = questionnairePageNumbers.filter(
      (number) => number !== pageNumber
    ).length

    setQuestionnaireSlots((prev) => ({
      ...prev,
      [pageNumber]: { file: null, preview: null },
    }))

    if (remainingCount === 0) {
      setQuestionnaireLockedUserId('')
      setQuestionnaireStatus(`已移除 Page ${pageNumber}；本次問卷學員鎖定已解除`)
    } else {
      setQuestionnaireStatus(`已移除 Page ${pageNumber}`)
    }
  }

  const submitQuestionnaire = async () => {
    if (questionnairePageCount === 0 || !selectedUser) {
      alert('請先拍攝至少一頁問卷並選擇學員！')
      return
    }

    // 送出時使用「第一張照片建立的鎖定學員」，避免 UI state 漂移造成綁錯人。
    const lockedTargetUserId = questionnaireLockedUserId || selectedUser

    if (lockedTargetUserId !== selectedUser) {
      setQuestionnaireStatus(
        '❌ 本次問卷鎖定的學員與目前選擇學員不同，已停止送出；請切換學員重新拍攝'
      )
      return
    }

    const targetUser = users.find((u) => u.id === lockedTargetUserId)
    const targetName = targetUser?.name || lockedTargetUserId
    const targetBranchId = selectedBranch
    const targetCoachId = authInfo?.coach_id

    if (!targetBranchId || !targetCoachId) {
      setQuestionnaireStatus(
        '❌ 尚未取得有效 branch_id 或 coach_id，請確認分店選擇與帳號設定'
      )
      alert('尚未取得有效 branch_id 或 coach_id，無法建立完整 HealthRecords')
      return
    }

    const pageText = questionnairePageNumbers.join('、')
    setQuestionnaireStatus(
      `🚀 正在辨識 Page ${pageText}（共 ${questionnairePageCount} 頁），分類 history / questionnaire 並與今日資料整合...`
    )
    setQuestionnaireResult(null)

    const data = new FormData()

    // 固定依 Page 1 → Page 4 順序送出；空 slot 不送。
    questionnairePageNumbers.forEach((pageNumber) => {
      data.append(
        'files',
        questionnaireSlots[pageNumber].file,
        `CoachOS_Page_${pageNumber}.jpg`
      )
    })
    // 將 slot 編號一併送到後端做 source event 紀錄；舊後端忽略此欄位也不影響。
    data.append('page_numbers', JSON.stringify(questionnairePageNumbers))

    data.append('target_user_id', lockedTargetUserId)
    data.append('user_name', targetName)
    data.append('company_id', authInfo.company_id)
    data.append('branch_id', targetBranchId)
    data.append('coach_id', targetCoachId)

    try {
      const token = await getIdToken()
      const response = await axios.post(
        `${CLOUD_API_URL}/api/analyze-questionnaire`,
        data,
        { headers: { Authorization: `Bearer ${token}` } }
      )

      console.log('問卷 API 完整回傳:', response.data)
      setQuestionnaireResult(response.data.ai_result)

      const mergeActions = response.data.merge_actions || []
      const actionText =
        mergeActions.length > 0 ? ` (${mergeActions.join(' / ')})` : ''

      // 成功後才清空 slot；失敗時保留照片，方便直接重試。
      revokeQuestionnaireSlotPreviews(questionnaireSlots)
      setQuestionnaireSlots(createEmptyQuestionnaireSlots())
      setQuestionnaireLockedUserId('')
      setQuestionnaireStatus(
        `🎉 辨識成功！已整合 Page ${pageText} 到今日 history / questionnaire${actionText}`
      )
    } catch (error) {
      console.error('Questionnaire API Error:', error)
      console.error('Questionnaire API Response:', error.response?.data)

      const status = error.response?.status
      const detail = error.response?.data?.detail

      if (status === 429) {
        setQuestionnaireStatus(
          '⚠️ AI 問卷辨識額度已達上限，請稍後再試，或聯繫場館管理員。已拍攝的問卷照片會保留。'
        )
      } else if (status === 422) {
        setQuestionnaireStatus(
          `❌ 問卷資料格式錯誤 (422)：${
            typeof detail === 'string' ? detail : JSON.stringify(detail)
          }`
        )
      } else if (status === 500) {
        setQuestionnaireStatus(
          `❌ 問卷解析失敗 (500)：${detail || '請查看 Render Logs'}；照片仍保留，可修正後重試`
        )
      } else {
        setQuestionnaireStatus(
          `❌ 問卷解析失敗 (${status || 'Network Error'})；照片仍保留，可重試`
        )
      }
    }
  }

  const submitToApi = async () => {
    if (!file || !selectedUser) {
      alert('請先拍照並選擇學員！')
      return
    }

    setUploadStatus('🚀 傳送至雲端 Gemini 解析中 (約需 10-15 秒)...')
    setLastResult(null)

    const targetUser = users.find((u) => u.id === selectedUser)
    const targetName = targetUser?.name || selectedUser
    const targetBranchId = selectedBranch
    const targetCoachId = authInfo?.coach_id

    // Upload identity uses authoritative coach capability + selected/fixed branch.
    // Backend will re-check Students association before writing HealthRecords.
    if (!targetBranchId || !targetCoachId) {
      setUploadStatus('❌ 尚未取得有效 branch_id 或 coach_id，請確認分店選擇與帳號設定')
      alert('尚未取得有效 branch_id 或 coach_id，無法建立完整 HealthRecords')
      return
    }

    const data = new FormData()
    data.append('file', file)
    data.append('target_user_id', selectedUser)
    data.append('user_name', targetName)
    data.append('company_id', authInfo.company_id)
    data.append('branch_id', targetBranchId)
    data.append('coach_id', targetCoachId)

    try {
      const token = await getIdToken()
      const response = await axios.post(
        `${CLOUD_API_URL}/api/analyze-inbody`,
        data,
        { headers: { Authorization: `Bearer ${token}` } }
      )

      console.log('API 完整回傳:', response.data)
      console.log('ai_result:', response.data.ai_result)

      setLastResult(response.data.ai_result)
      setUploadStatus('🎉 雲端解析成功！已存入 HealthRecords')
      setFile(null)
      setPreview(null)
    } catch (error) {
      console.error('API Error:', error)
      console.error('API Response:', error.response?.data)

      const status = error.response?.status

      if (status === 429) {
        setUploadStatus(
          '⚠️ AI 辨識額度已達上限，本次未完成辨識。請稍後再試，或聯繫場館管理員。已選擇的照片會保留。'
        )
      } else if (status === 422) {
        const detail = error.response?.data?.detail
        setUploadStatus(
          `❌ API 資料格式錯誤 (422)：${
            typeof detail === 'string' ? detail : JSON.stringify(detail)
          }`
        )
      } else if (status === 504) {
        setUploadStatus('😫 伺服器反應過久，請再試一次')
      } else if (status === 500) {
        setUploadStatus(
          `❌ 雲端解析失敗 (500)：${
            error.response?.data?.detail || '請查看 Render Logs'
          }`
        )
      } else {
        setUploadStatus(
          `❌ 解析失敗 (${status || 'Network Error'})，請檢查 API 日誌`
        )
      }
    }
  }

  const measurements = lastResult?.measurements || {}
  const segmental = lastResult?.segmental_muscle || {}
  const derived = lastResult?.derived_metrics || {}
  const scores = lastResult?.scores || {}
  const coachos = scores?.coachos || {}
  const dataQuality = lastResult?.data_quality || {}

  const heightM =
    measurements?.height_m ??
    lastResult?.height_m ??
    null

  const weightKg =
    measurements?.weight_kg ??
    lastResult?.weight_kg ??
    null

  const bodyFatPct =
    measurements?.body_fat_percentage ??
    lastResult?.body_fat_percentage ??
    null

  const visceralFat =
    measurements?.visceral_fat_value ??
    lastResult?.visceral_fat_level ??
    null

  const reportedSmi =
    measurements?.smi_kg_m2 ??
    null

  const genderRaw = lastResult?.gender ?? null
  const genderText =
    genderRaw === 'male'
      ? '男性'
      : genderRaw === 'female'
        ? '女性'
        : genderRaw || '--'

  const displayValue = (value, suffix = '') => {
    if (value === null || value === undefined || value === '') return '--'
    return `${value}${suffix}`
  }

  const genderLabel = (value) => {
    if (value === 'male') return '男性'
    if (value === 'female') return '女性'
    return value || '--'
  }

  const listText = (value) => {
    if (!Array.isArray(value) || value.length === 0) return '--'
    return value
      .map((item) => {
        if (typeof item === 'string') return item
        if (item?.procedure_or_body_part) {
          const details = [
            item.procedure_or_body_part,
            item.year_or_date,
            item.current_effect_or_limitation,
          ].filter(Boolean)
          return details.join(' / ')
        }
        if (item?.name || item?.category) {
          return [item.category, item.name, item.note].filter(Boolean).join(' / ')
        }
        return JSON.stringify(item)
      })
      .join('、')
  }

  const scoreStatusText = (status) => {
    if (status === 'complete') return '完整'
    if (status === 'partial') return '部分資料'
    if (status === 'insufficient_gender') return '缺少性別'
    if (status === 'insufficient_data') return '資料不足'
    return status || '--'
  }

  return (
    <div className="scanner-app">
      {authLoading ? (
        <div className="scanner-state-panel">
          <span className="scanner-spinner" aria-hidden="true" />
          <strong>正在確認登入狀態</strong>
          <small>正在連線至 CoachOS 帳號與 Scanner API</small>
        </div>
      ) : authError ? (
        <div className="scanner-error-card">
          <div className="scanner-error-icon">!</div>
          <div>
            <h3>Scanner 無法完成後端驗證</h3>
            <p>{authError}</p>
            <p className="scanner-error-help">
              這不代表 Portal 登入失效。若為本機整合測試，請確認兩支 API 都允許
              http://127.0.0.1:5173。
            </p>
            <div className="scanner-error-actions">
              <button className="scanner-primary-small" onClick={() => bootstrapSharedLogin()}>重新驗證</button>
              <button className="scanner-ghost-small" onClick={() => window.location.assign('/')}>回入口</button>
            </div>
          </div>
        </div>
      ) : !isLoggedIn || !authInfo ? (
        <div className="scanner-state-panel">
          <span className="scanner-spinner" aria-hidden="true" />
          <strong>登入狀態無效</strong>
          <small>正在返回動康評 | CoachOS 入口</small>
        </div>
      ) : (
        <div className="scanner-shell">
          <header className="scanner-topbar">
            <button className="scanner-brand" type="button" onClick={() => window.location.assign('/')} aria-label="回到入口">
              <span className="scanner-brand-mark">V</span>
              <span className="scanner-brand-copy">
                <strong>動康評 <em>| CoachOS</em></strong>
                <small>Vitruvian Sport Technology</small>
              </span>
            </button>
            <div className="scanner-topbar-actions">
              <button className="scanner-ghost-small" onClick={() => window.location.assign('/')}>入口</button>
              <button className="scanner-text-danger" onClick={handleLogout}>登出</button>
            </div>
          </header>

          <section className="scanner-intro">
            <span className="scanner-eyebrow">DATA CAPTURE</span>
            <h1>智慧掃描端</h1>
            <p>快速完成身體組成與健康問卷資料收集，直接銜接 CoachOS。</p>
          </section>

          <section className="scanner-context-card">
            <div className="scanner-context-item"><span>企業</span><strong>{authInfo.company_id}</strong></div>
            <div className="scanner-context-item"><span>教練</span><strong>{authInfo.coach_id}</strong></div>
            <div className="scanner-context-item"><span>權限</span><strong>{authInfo.role}</strong></div>
            <div className="scanner-context-item scanner-context-wide">
              <span>分店</span>
              <strong>{authInfo.role === 'owner' ? selectedBranch || '請先選擇分店' : authInfo.branch_id || '--'}</strong>
            </div>
          </section>

          {authInfo.role === 'owner' && (
            <section className="scanner-owner-card">
              <label htmlFor="owner-branch-select" className="scanner-field-label">
                📍 Owner 請先選擇本次操作分店
              </label>
              <select
                id="owner-branch-select"
                value={selectedBranch}
                onChange={handleBranchChange}
                className="scanner-select" 
              >
                <option value="">請選擇分店</option>
                {branches.map((branch) => (
                  <option key={branch.id} value={branch.id}>
                    {branch.name} ({branch.id})
                  </option>
                ))}
              </select>
              {selectedBranch && users.length === 0 && (
                <p className="scanner-help-text">
                  此分店目前沒有與 {authInfo.coach_id} 建立 Students 關聯的學員。
                </p>
              )}
            </section>
          )}

          <div className="scanner-mode-switch">
            <button
              onClick={() => setScanMode('body')}
              className={`scanner-mode-button ${scanMode === 'body' ? 'active' : ''}`}
            >
              <span className="scanner-mode-icon">◎</span> 身體組成
            </button>

            <button
              onClick={() => setScanMode('questionnaire')}
              className={`scanner-mode-button ${scanMode === 'questionnaire' ? 'active' : ''}`}
            >
              <span className="scanner-mode-icon">▤</span> 健康問卷
            </button>
          </div>

          {scanMode === 'body' && lastResult && (
            <section className="scanner-result-card body-result-card">
              <div className="scanner-result-heading">
                <span className="scanner-result-icon">✓</span>
                <div><small>BODY COMPOSITION</small><h3>雲端高階分析結果</h3></div>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '10px',
                  marginBottom: '15px',
                  fontSize: '15px',
                }}
              >
                <div>
                  <p style={{ margin: '6px 0' }}>
                    <strong>設備:</strong>{' '}
                    <span style={{ color: '#666' }}>
                      {lastResult.device_type?.toUpperCase() || '--'}
                    </span>
                  </p>

                  <p style={{ margin: '6px 0' }}>
                    <strong>性別:</strong> {genderText}
                  </p>

                  <p style={{ margin: '6px 0' }}>
                    <strong>身高:</strong> {displayValue(heightM, ' m')}
                  </p>

                  <p style={{ margin: '6px 0' }}>
                    <strong>體重:</strong> {displayValue(weightKg, ' kg')}
                  </p>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <p style={{ margin: '6px 0' }}>
                    <strong>體脂率:</strong>{' '}
                    <span
                      style={{
                        color: '#D81B60',
                        fontWeight: 'bold',
                      }}
                    >
                      {displayValue(bodyFatPct, '%')}
                    </span>
                  </p>

                  <p style={{ margin: '6px 0' }}>
                    <strong>內臟脂肪:</strong>{' '}
                    {displayValue(visceralFat)}
                  </p>

                </div>
              </div>

              <div
                style={{
                  background: '#EFFBF9',
                  padding: '12px',
                  borderRadius: '8px',
                  marginBottom: '15px',
                  border: '1px solid #D5EFEB',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    gap: '10px',
                  }}
                >
                  <strong style={{ color: '#087C77' }}>
                    CoachOS 體組成分數
                  </strong>

                  <span
                    style={{
                      fontSize: '26px',
                      fontWeight: 'bold',
                      color: '#087C77',
                    }}
                  >
                    {coachos?.value ?? '--'}
                  </span>
                </div>

                <p
                  style={{
                    margin: '6px 0 0 0',
                    fontSize: '12px',
                    color: '#666',
                  }}
                >
                  {coachos?.version || '--'} ・{' '}
                  {scoreStatusText(coachos?.status)}
                  {typeof coachos?.coverage === 'number'
                    ? ` ・ Coverage ${(coachos.coverage * 100).toFixed(0)}%`
                    : ''}
                </p>

                {coachos?.components && (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(2, 1fr)',
                      gap: '6px',
                      marginTop: '10px',
                      textAlign: 'center',
                      fontSize: '12px',
                    }}
                  >
                    <div>
                      肌肉
                      <br />
                      <strong>{coachos.components.muscle ?? '--'}</strong>
                    </div>

                    <div>
                      體脂
                      <br />
                      <strong>{coachos.components.body_fat ?? '--'}</strong>
                    </div>

                  </div>
                )}

                {scores?.inbody_score !== null &&
                  scores?.inbody_score !== undefined && (
                    <p
                      style={{
                        margin: '10px 0 0 0',
                        fontSize: '13px',
                        color: '#555',
                      }}
                    >
                      InBody Score：<strong>{scores.inbody_score}</strong>
                    </p>
                  )}
              </div>

              <div
                style={{
                  background: '#F3FBFA',
                  padding: '12px',
                  borderRadius: '8px',
                  marginBottom: '15px',
                  fontSize: '15px',
                }}
              >
                <p style={{ margin: '6px 0' }}>
                  🔹 <strong>SMI (肌質量指數):</strong>{' '}
                  <span
                    style={{
                      color: '#087C77',
                      fontWeight: 'bold',
                      fontSize: '16px',
                    }}
                  >
                    {derived?.SMI ?? '--'}
                  </span>
                </p>

                <p style={{ margin: '6px 0' }}>
                  🔹 <strong>ALM (四肢肌肉量):</strong>{' '}
                  {derived?.ALM ?? '--'} kg
                </p>

                {reportedSmi !== null &&
                  derived?.SMI !== null &&
                  derived?.SMI !== undefined &&
                  Math.abs(Number(reportedSmi) - Number(derived.SMI)) > 0.02 && (
                    <p
                      style={{
                        margin: '6px 0',
                        fontSize: '12px',
                        color: '#666',
                      }}
                    >
                      報表 SMI：{reportedSmi}
                    </p>
                  )}
              </div>

              <div style={{ marginBottom: '15px' }}>
                <p
                  style={{
                    fontWeight: 'bold',
                    marginBottom: '8px',
                    color: '#0DAEA1',
                    fontSize: '15px',
                  }}
                >
                  💪 節段肌肉分布 (kg)
                </p>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '8px',
                    border: '1px solid #EFFBF9',
                    padding: '10px',
                    borderRadius: '6px',
                    fontSize: '15px',
                    textAlign: 'center',
                    background: '#FAFAFA',
                  }}
                >
                  <div style={{ borderRight: '1px solid #ddd' }}>
                    左上:
                    <br />
                    <strong style={{ color: '#087C77' }}>
                      {segmental?.la ?? '--'}
                    </strong>
                  </div>

                  <div>
                    右上:
                    <br />
                    <strong style={{ color: '#087C77' }}>
                      {segmental?.ra ?? '--'}
                    </strong>
                  </div>

                  <div
                    style={{
                      borderRight: '1px solid #ddd',
                      borderTop: '1px solid #ddd',
                      paddingTop: '8px',
                    }}
                  >
                    左下:
                    <br />
                    <strong style={{ color: '#087C77' }}>
                      {segmental?.ll ?? '--'}
                    </strong>
                  </div>

                  <div
                    style={{
                      borderTop: '1px solid #ddd',
                      paddingTop: '8px',
                    }}
                  >
                    右下:
                    <br />
                    <strong style={{ color: '#087C77' }}>
                      {segmental?.rl ?? '--'}
                    </strong>
                  </div>
                </div>
              </div>

              <div>
                <p
                  style={{
                    fontWeight: 'bold',
                    marginBottom: '8px',
                    color: '#E65100',
                    fontSize: '15px',
                  }}
                >
                  ⚠️ 左右不平衡指數 (AI %)
                </p>

                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '10px 12px',
                    background: '#FFF3E0',
                    borderRadius: '6px',
                    fontSize: '15px',
                    gap: '8px',
                  }}
                >
                  <span>
                    上肢失衡:{' '}
                    <strong
                      style={{
                        color:
                          derived?.AI_upper_pct > 10
                            ? '#d32f2f'
                            : '#E65100',
                      }}
                    >
                      {derived?.AI_upper_pct ?? '--'}%
                    </strong>
                  </span>

                  <span>
                    下肢失衡:{' '}
                    <strong
                      style={{
                        color:
                          derived?.AI_lower_pct > 10
                            ? '#d32f2f'
                            : '#E65100',
                      }}
                    >
                      {derived?.AI_lower_pct ?? '--'}%
                    </strong>
                  </span>
                </div>

                {(derived?.AI_upper_pct > 10 ||
                  derived?.AI_lower_pct > 10) && (
                  <p
                    style={{
                      color: '#d32f2f',
                      fontSize: '13px',
                      marginTop: '8px',
                      fontWeight: 'bold',
                    }}
                  >
                    * 指數 &gt; 10% 建議加強對側穩定訓練
                  </p>
                )}
              </div>

              {dataQuality?.validation_warnings?.length > 0 && (
                <div
                  style={{
                    marginTop: '15px',
                    background: '#FFF8E1',
                    border: '1px solid #FFE082',
                    padding: '10px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    color: '#795548',
                  }}
                >
                  <strong>資料品質提醒：</strong>
                  <ul style={{ margin: '6px 0 0 18px', padding: 0 }}>
                    {dataQuality.validation_warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              <p
                style={{
                  fontSize: '12px',
                  color: '#999',
                  marginTop: '15px',
                  textAlign: 'center',
                }}
              >
                解析時間: {lastResult.createdAt || '--'}
              </p>
            </section>
          )}

          {scanMode === 'questionnaire' && questionnaireResult && (
            <section className="scanner-result-card questionnaire-result-card">
              <div className="scanner-result-heading">
                <span className="scanner-result-icon">✓</span>
                <div><small>HEALTH QUESTIONNAIRE</small><h3>今日問卷辨識結果</h3></div>
              </div>

              <div
                style={{
                  background: '#F3FBFA',
                  padding: '10px',
                  borderRadius: '8px',
                  marginBottom: '10px',
                  fontSize: '14px',
                }}
              >
                <strong>來源：</strong> CoachOS 標準問卷（紙本 / 手機拍照）
                <br />
                <strong>姓名：</strong>{' '}
                {questionnaireResult.profile?.name_on_form ||
                  questionnaireResult.user_name ||
                  '--'}
                <br />
                <strong>年齡：</strong>{' '}
                {questionnaireResult.profile?.age ?? '--'}
                <br />
                <strong>性別：</strong>{' '}
                {genderLabel(questionnaireResult.profile?.gender)}
              </div>

              <div
                style={{
                  background: '#F9F9F9',
                  border: '1px solid #E0E0E0',
                  borderRadius: '8px',
                  padding: '10px',
                  marginBottom: '12px',
                  fontSize: '12px',
                  color: '#555',
                  wordBreak: 'break-all',
                }}
              >
                <strong>今日 Firebase 整合：</strong>
                <br />
                History：{' '}
                {questionnaireResult.saved_documents?.history
                  ? `${questionnaireResult.saved_documents.history.doc_id}（累計 ${questionnaireResult.saved_documents.history.upload_count ?? 1} 次上傳）`
                  : '本次沒有疾病 / 手術史內容'}
                <br />
                Questionnaire：{' '}
                {questionnaireResult.saved_documents?.questionnaire
                  ? `${questionnaireResult.saved_documents.questionnaire.doc_id}（累計 ${questionnaireResult.saved_documents.questionnaire.upload_count ?? 1} 次上傳）`
                  : '本次沒有生活 / 運動問卷內容'}
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong style={{ color: '#C62828' }}>
                  ⚠️ 已勾選健康 / 安全資訊
                </strong>
                <p style={{ margin: '6px 0', fontSize: '14px', lineHeight: 1.6 }}>
                  {listText(questionnaireResult.display?.history_selected_labels)}
                </p>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <strong style={{ color: '#087C77' }}>
                  🎯 已勾選生活 / 運動偏好
                </strong>
                <p style={{ margin: '6px 0', fontSize: '14px', lineHeight: 1.6 }}>
                  {listText(questionnaireResult.display?.questionnaire_selected_labels)}
                </p>
              </div>

              {questionnaireResult.data_quality?.review_required && (
                <div
                  style={{
                    background: '#FFF8E1',
                    border: '1px solid #FFE082',
                    padding: '10px',
                    borderRadius: '6px',
                    fontSize: '12px',
                    color: '#795548',
                  }}
                >
                  <strong>需要人工確認：</strong>
                  <div>
                    {listText([
                      ...(questionnaireResult.data_quality?.validation_warnings || []),
                      ...(questionnaireResult.data_quality?.unreadable_or_ambiguous_items || []),
                      ...(questionnaireResult.data_quality?.merge_conflicts || []).map(
                        (item) =>
                          `補拍資料差異：${item.field}（採用最新辨識值）`
                      ),
                    ])}
                  </div>
                </div>
              )}
            </section>
          )}

          <section className="scanner-student-card">
            <div className="scanner-section-heading">
              <div><small>MEMBER</small><strong>選擇學員</strong></div>
              <span>Step 1</span>
            </div>

            <select
              value={selectedUser}
              onChange={handleSelectedUserChange}
              className={`scanner-select scanner-member-select ${questionnaireLockedUserId ? 'locked' : ''}`}
            >
              {users.map((u) => (
                <option key={`${u.id}-${u.branch_id}-${u.coach_id}`} value={u.id}>
                  {u.name} ({String(u.id).substring(0, 5)}...)
                </option>
              ))}
            </select>

            {scanMode === 'questionnaire' && questionnaireLockedUserId && (
              <div className="scanner-lock-note">
                <span>🔒</span> 本次未送出問卷已鎖定此學員；若切換學員，Page 1～4 尚未送出的照片會自動清空。
              </div>
            )}
          </section>

          {scanMode === 'body' && (
            <>
              <section className="scanner-capture-card">
                <div className="scanner-section-heading">
                  <div><small>BODY COMPOSITION</small><strong>拍攝 InBody / 體組成報表</strong></div>
                  <span>Step 2</span>
                </div>
                <label htmlFor="camera-input" className="scanner-capture-button">
                  <span className="scanner-capture-icon">＋</span>
                  <strong>開啟相機拍照</strong>
                  <small>請完整拍攝報表，避免反光與裁切</small>
                </label>

            <input
              id="camera-input"
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleImageUpload}
              style={{ display: 'none' }}
            />

                {preview && (
                  <img src={preview} alt="預覽" className="scanner-preview-image" />
                )}

                <p className="scanner-status-text">{uploadStatus}</p>

                <button onClick={submitToApi} className="scanner-primary-action">
                  傳送並解析
                </button>
              </section>
            </>
          )}

          {scanMode === 'questionnaire' && (
            <>
              <section className="scanner-capture-card questionnaire-capture-card">
                <div className="scanner-section-heading">
                  <div><small>HEALTH QUESTIONNAIRE</small><strong>健康問卷拍照</strong></div>
                  <span>Step 2</span>
                </div>
                <div className="scanner-info-panel">
                  <strong>Page Slot 防呆</strong>
                  <p>請把照片放進正確的 Page 1～Page 4。第一張照片拍下後會鎖定目前學員；切換學員時，尚未送出的照片會自動清空。同一 Page 重拍會直接取代舊照片。空白 Page 不送出，也可只補拍其中幾頁。</p>
                </div>

              <div className="questionnaire-grid">
                {QUESTIONNAIRE_PAGE_NUMBERS.map((pageNumber) => {
                  const slot = questionnaireSlots[pageNumber]
                  const hasPhoto = Boolean(slot?.file)

                  return (
                    <div key={pageNumber} className={`questionnaire-slot ${hasPhoto ? 'has-photo' : ''}`}>
                      <div className="questionnaire-slot-head">
                        <span>Page {pageNumber}</span>
                        <small>{hasPhoto ? '● 已拍攝' : '尚未拍攝'}</small>
                      </div>

                      {slot?.preview ? (
                        <img
                          src={slot.preview}
                          alt={`CoachOS 問卷 Page ${pageNumber}`}
                          className="questionnaire-slot-preview" 
                        />
                      ) : (
                        <div className="questionnaire-slot-empty">
                          <span>＋</span>
                          <small>請拍攝第 {pageNumber} 頁</small>
                        </div>
                      )}

                      <div className="questionnaire-slot-actions">
                        <label
                          htmlFor={`questionnaire-camera-input-${pageNumber}`}
                          className="questionnaire-photo-button" 
                        >
                          {hasPhoto ? `🔄 重拍 Page ${pageNumber}` : `📷 拍 Page ${pageNumber}`}
                        </label>

                        <input
                          id={`questionnaire-camera-input-${pageNumber}`}
                          type="file"
                          accept="image/*"
                          capture="environment"
                          onChange={(event) =>
                            handleQuestionnaireImageUpload(pageNumber, event)
                          }
                          style={{ display: 'none' }}
                        />

                        {hasPhoto && (
                          <button
                            type="button"
                            onClick={() => removeQuestionnairePage(pageNumber)}
                            className="questionnaire-remove-button" 
                          >
                            移除
                          </button>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>

              {questionnairePageCount > 0 && (
                <div className="questionnaire-pending-summary">
                  <strong>本次待送出：</strong> Page{' '}
                  {questionnairePageNumbers.join('、')}（共 {questionnairePageCount} 頁）
                </div>
              )}

              <p className="scanner-status-text">{questionnaireStatus}</p>

              <button
                onClick={submitQuestionnaire}
                disabled={questionnairePageCount === 0}
                className="scanner-primary-action" 
              >
                辨識並整合 ({questionnairePageCount} 頁)
              </button>

              {questionnairePageCount > 0 && (
                <button
                  type="button"
                  onClick={() =>
                    clearQuestionnaireDraft('已清空本次未送出的問卷照片，學員鎖定已解除')
                  }
                  className="scanner-secondary-action" 
                >
                  清空 Page 1～4 未送出照片
                </button>
              )}
              </section>
            </>
          )}

        </div>
      )}
    </div>
  )
}

export default App
