import { useState, useEffect } from 'react'
import imageCompression from 'browser-image-compression'
import axios from 'axios'
import './App.css'

const FIREBASE_API_KEY = import.meta.env.VITE_FIREBASE_API_KEY || ''
const CLOUD_API_URL = import.meta.env.VITE_CLOUD_API_URL || 'http://127.0.0.1:8000'
const FIREBASE_SESSION_KEY = 'firebaseSession'

const safeJsonParse = (value, fallback = null) => {
  try {
    return value ? JSON.parse(value) : fallback
  } catch {
    return fallback
  }
}

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
  const [isLoggedIn, setIsLoggedIn] = useState(
    localStorage.getItem('isLoggedIn') === 'true' &&
      Boolean(localStorage.getItem(FIREBASE_SESSION_KEY))
  )
  const [authInfo, setAuthInfo] = useState(
    safeJsonParse(localStorage.getItem('authInfo'), null)
  )

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loginStatus, setLoginStatus] = useState('')

  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState('')
  const [branches, setBranches] = useState([])
  const [selectedBranch, setSelectedBranch] = useState(
    safeJsonParse(localStorage.getItem('authInfo'), null)?.role === 'owner'
      ? ''
      : safeJsonParse(localStorage.getItem('authInfo'), null)?.branch_id || ''
  )

  // Muzili 身體組成：人工基本資料 + 兩個固定圖片 slot
  const [muziliHeightCm, setMuziliHeightCm] = useState('')
  const [muziliAge, setMuziliAge] = useState('')
  const [muziliGender, setMuziliGender] = useState('')
  const [muziliSlots, setMuziliSlots] = useState({
    bodyData: { file: null, preview: null },
    segmentalMuscle: { file: null, preview: null },
  })
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

  const saveFirebaseSession = (session) => {
    localStorage.setItem(FIREBASE_SESSION_KEY, JSON.stringify(session))
  }

  const getIdToken = async () => {
    const session = safeJsonParse(localStorage.getItem(FIREBASE_SESSION_KEY), null)
    if (!session?.idToken) {
      throw new Error('登入憑證不存在，請重新登入')
    }

    const expiresAt = Number(session.expiresAt || 0)
    if (expiresAt > Date.now() + 60_000) {
      return session.idToken
    }

    if (!session.refreshToken) {
      throw new Error('登入憑證已過期，請重新登入')
    }

    const refreshUrl =
      `https://securetoken.googleapis.com/v1/token?key=${FIREBASE_API_KEY}`
    const params = new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: session.refreshToken,
    })

    const response = await axios.post(refreshUrl, params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })

    const refreshed = {
      idToken: response.data.id_token,
      refreshToken: response.data.refresh_token || session.refreshToken,
      expiresAt:
        Date.now() + Number(response.data.expires_in || 3600) * 1000,
    }
    saveFirebaseSession(refreshed)
    return refreshed.idToken
  }

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

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoginStatus('🔄 驗證中...')

    try {
      const url =
        `https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=${FIREBASE_API_KEY}`

      const response = await axios.post(url, {
        email,
        password,
        returnSecureToken: true,
      })

      const token = response.data.idToken
      const session = {
        idToken: token,
        refreshToken: response.data.refreshToken,
        expiresAt:
          Date.now() + Number(response.data.expiresIn || 3600) * 1000,
      }
      saveFirebaseSession(session)

      // Do not trust decoded custom claims for authorization.
      // Backend /api/me resolves the authoritative Coaches/{uid} document.
      const meResponse = await axios.get(`${CLOUD_API_URL}/api/me`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      const actor = meResponse.data.actor || {}
      const roles = roleCapabilities(actor.role, actor.roles)

      if (!roles.includes('coach')) {
        throw new Error('此帳號沒有 coach 上傳屬性')
      }

      const newAuthInfo = {
        company_id: actor.company_id,
        branch_id: actor.branch_id || null,
        coach_id: actor.coach_id,
        role: actor.role,
        roles,
      }

      localStorage.setItem('isLoggedIn', 'true')
      localStorage.setItem('authInfo', JSON.stringify(newAuthInfo))

      setAuthInfo(newAuthInfo)
      setIsLoggedIn(true)
      setSelectedBranch(newAuthInfo.role === 'owner' ? '' : newAuthInfo.branch_id || '')
      setLoginStatus('')
    } catch (error) {
      console.error('Login Error:', error)
      localStorage.removeItem('isLoggedIn')
      localStorage.removeItem('authInfo')
      localStorage.removeItem(FIREBASE_SESSION_KEY)
      const firebaseMessage =
        error.response?.data?.error?.message ||
        error.response?.data?.detail ||
        error.message ||
        '請檢查帳號密碼或系統設定'
      setLoginStatus(`❌ 登入失敗：${firebaseMessage}`)
    }
  }

  const revokeMuziliSlotPreviews = (slots) => {
    Object.values(slots || {}).forEach((slot) => {
      if (slot?.preview) URL.revokeObjectURL(slot.preview)
    })
  }

  const resetMuziliSlots = () => ({
    bodyData: { file: null, preview: null },
    segmentalMuscle: { file: null, preview: null },
  })

  const clearMuziliDraft = (statusMessage = '等待操作') => {
    revokeMuziliSlotPreviews(muziliSlots)
    setMuziliSlots(resetMuziliSlots())
    setMuziliHeightCm('')
    setMuziliAge('')
    setMuziliGender('')
    setUploadStatus(statusMessage)
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

  const handleLogout = () => {
    localStorage.removeItem('isLoggedIn')
    localStorage.removeItem('authInfo')
    localStorage.removeItem(FIREBASE_SESSION_KEY)

    setIsLoggedIn(false)
    setAuthInfo(null)
    setBranches([])
    setSelectedBranch('')
    setUsers([])
    setSelectedUser('')

    revokeMuziliSlotPreviews(muziliSlots)
    setMuziliSlots(resetMuziliSlots())
    setMuziliHeightCm('')
    setMuziliAge('')
    setMuziliGender('')
    setLastResult(null)
    setUploadStatus('等待操作')
    setScanMode('body')

    revokeQuestionnaireSlotPreviews(questionnaireSlots)
    setQuestionnaireSlots(createEmptyQuestionnaireSlots())
    setQuestionnaireLockedUserId('')
    setQuestionnaireResult(null)
    setQuestionnaireStatus('等待操作')
  }

  const handleMuziliImageUpload = async (slotKey, event) => {
    const imageFile = event.target.files?.[0]
    if (!imageFile) return

    if (!selectedUser) {
      alert('請先選擇學員！')
      event.target.value = ''
      return
    }

    if (!['bodyData', 'segmentalMuscle'].includes(slotKey)) {
      setUploadStatus('❌ 未知的 Muzili 圖片欄位')
      event.target.value = ''
      return
    }

    const slotLabel = slotKey === 'bodyData' ? '身體數據' : '節段肌肉分析'
    setUploadStatus(`🔄 ${slotLabel}圖片處理中...`)

    try {
      const compressedFile = await imageCompression(imageFile, {
        maxSizeMB: 1.2,
        maxWidthOrHeight: 1800,
        useWebWorker: true,
      })
      const previewUrl = URL.createObjectURL(compressedFile)

      setMuziliSlots((prev) => {
        const oldPreview = prev[slotKey]?.preview
        if (oldPreview) URL.revokeObjectURL(oldPreview)

        return {
          ...prev,
          [slotKey]: {
            file: compressedFile,
            preview: previewUrl,
          },
        }
      })

      setUploadStatus(`✅ ${slotLabel}圖片已選取`)
      event.target.value = ''
    } catch (error) {
      console.error('Muzili 圖片處理失敗:', error)
      setUploadStatus(`❌ ${slotLabel}圖片處理失敗`)
      event.target.value = ''
    }
  }

  const handleBranchChange = (event) => {
    const nextBranch = event.target.value
    if (nextBranch === selectedBranch) return

    const hadPendingQuestionnaire = questionnairePageCount > 0
    const hadPendingBodyData =
      Boolean(muziliSlots.bodyData?.file) ||
      Boolean(muziliSlots.segmentalMuscle?.file) ||
      Boolean(muziliHeightCm) ||
      Boolean(muziliAge) ||
      Boolean(muziliGender)

    clearMuziliDraft(
      hadPendingBodyData
        ? '⚠️ 已切換分店，未送出的 Muzili 身體組成資料已自動清空'
        : '等待操作'
    )

    clearQuestionnaireDraft(
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
    const hadPendingBodyData =
      Boolean(muziliSlots.bodyData?.file) ||
      Boolean(muziliSlots.segmentalMuscle?.file) ||
      Boolean(muziliHeightCm) ||
      Boolean(muziliAge) ||
      Boolean(muziliGender)

    // 切換學員時，清空所有尚未送出的 Muzili / 問卷內容，避免 A 的資料被送到 B。
    clearMuziliDraft(
      hadPendingBodyData
        ? '⚠️ 已切換學員，未送出的 Muzili 身體組成資料已自動清空'
        : '等待操作'
    )

    clearQuestionnaireDraft(
      hadPendingQuestionnaire
        ? '⚠️ 已切換學員，未送出的問卷照片已自動清空'
        : '等待操作'
    )

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

      if (status === 422) {
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

  const submitMuziliToApi = async () => {
    if (!selectedUser) {
      alert('請先選擇學員！')
      return
    }

    const heightCm = Number(muziliHeightCm)
    const age = Number(muziliAge)

    if (!Number.isFinite(heightCm) || heightCm < 100 || heightCm > 250) {
      alert('請輸入正確身高，例如 168.0 cm')
      return
    }

    if (!Number.isInteger(age) || age < 1 || age > 120) {
      alert('請輸入正確實際年齡，例如 42 歲')
      return
    }

    if (!['male', 'female'].includes(muziliGender)) {
      alert('請選擇性別')
      return
    }

    if (!muziliSlots.bodyData?.file) {
      alert('請上傳 Muzili「身體數據」圖片')
      return
    }

    if (!muziliSlots.segmentalMuscle?.file) {
      alert('請上傳「節段分析 → 肌肉量」圖片')
      return
    }

    const targetUser = users.find((u) => u.id === selectedUser)
    const targetName = targetUser?.name || selectedUser
    const targetBranchId = selectedBranch
    const targetCoachId = authInfo?.coach_id

    if (!targetBranchId || !targetCoachId) {
      setUploadStatus('❌ 尚未取得有效 branch_id 或 coach_id，請確認分店與帳號設定')
      return
    }

    setUploadStatus('🚀 Muzili 身體組成資料送至雲端辨識中...')
    setLastResult(null)

    const data = new FormData()
    data.append('body_data_file', muziliSlots.bodyData.file, 'muzili_body_data.jpg')
    data.append(
      'segmental_muscle_file',
      muziliSlots.segmentalMuscle.file,
      'muzili_segmental_muscle.jpg'
    )
    data.append('height_cm', String(heightCm))
    data.append('age', String(age))
    data.append('gender', muziliGender)
    data.append('target_user_id', selectedUser)
    data.append('user_name', targetName)
    data.append('company_id', authInfo.company_id)
    data.append('branch_id', targetBranchId)
    data.append('coach_id', targetCoachId)

    try {
      const token = await getIdToken()
      const response = await axios.post(
        `${CLOUD_API_URL}/api/analyze-muzili-body-composition`,
        data,
        { headers: { Authorization: `Bearer ${token}` } }
      )

      console.log('Muzili API 完整回傳:', response.data)
      setLastResult(response.data.ai_result)

      // 成功後清空圖片，保留人工輸入的 height / age / gender 供使用者核對。
      revokeMuziliSlotPreviews(muziliSlots)
      setMuziliSlots(resetMuziliSlots())
      setUploadStatus('🎉 Muzili 身體組成解析成功！已存入 HealthRecords')
    } catch (error) {
      console.error('Muzili API Error:', error)
      console.error('Muzili API Response:', error.response?.data)

      const status = error.response?.status
      const detail = error.response?.data?.detail

      if (status === 422) {
        setUploadStatus(
          `❌ Muzili 資料格式錯誤：${
            typeof detail === 'string' ? detail : JSON.stringify(detail)
          }`
        )
      } else if (status === 504) {
        setUploadStatus('😫 伺服器反應過久，請再試一次')
      } else if (status === 500) {
        setUploadStatus(`❌ Muzili 雲端解析失敗：${detail || '請查看 Render Logs'}`)
      } else {
        setUploadStatus(`❌ 解析失敗 (${status || 'Network Error'})`)
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
    genderRaw === 'male' || genderRaw === 'm'
      ? '男性'
      : genderRaw === 'female' || genderRaw === 'f'
        ? '女性'
        : genderRaw || '--'

  const displayValue = (value, suffix = '') => {
    if (value === null || value === undefined || value === '') return '--'
    return `${value}${suffix}`
  }

  const genderLabel = (value) => {
    if (value === 'male' || value === 'm') return '男性'
    if (value === 'female' || value === 'f') return '女性'
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
    <div
      style={{
        padding: '20px',
        maxWidth: '420px',
        margin: '0 auto',
        fontFamily: 'sans-serif',
      }}
    >
      {!isLoggedIn ? (
        <form
          onSubmit={handleLogin}
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '15px',
          }}
        >
          <h2>🔐 企業教練登入</h2>

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={{ padding: '10px' }}
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ padding: '10px' }}
          />

          <button
            type="submit"
            style={{
              padding: '12px',
              background: '#2196F3',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              fontSize: '16px',
            }}
          >
            登入
          </button>

          <p style={{ color: 'red' }}>{loginStatus}</p>
        </form>
      ) : (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '15px',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <h2 style={{ margin: 0 }}>📱 智慧掃描端</h2>

            <button
              onClick={handleLogout}
              style={{
                padding: '6px 12px',
                background: '#f44336',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
              }}
            >
              登出
            </button>
          </div>

          <div
            style={{
              background: '#f5f5f5',
              padding: '12px',
              borderRadius: '8px',
              fontSize: '14px',
              textAlign: 'left',
              border: '1px solid #ddd',
              color: '#666',
            }}
          >
            <strong>🏢 企業:</strong> {authInfo.company_id}
            <br />
            <strong>🧑‍🏫 教練:</strong> {authInfo.coach_id}
            <br />
            <strong>🔐 權限:</strong> {authInfo.role}
            <br />
            <strong>📍 分店:</strong>{' '}
            {authInfo.role === 'owner'
              ? selectedBranch || '請先選擇分店'
              : authInfo.branch_id || '--'}
          </div>

          {authInfo.role === 'owner' && (
            <div
              style={{
                background: '#FFF8E1',
                padding: '12px',
                borderRadius: '8px',
                border: '1px solid #FFE082',
              }}
            >
              <label
                htmlFor="owner-branch-select"
                style={{
                  display: 'block',
                  fontWeight: 'bold',
                  marginBottom: '6px',
                  color: '#7A5200',
                }}
              >
                📍 Owner 請先選擇本次操作分店
              </label>
              <select
                id="owner-branch-select"
                value={selectedBranch}
                onChange={handleBranchChange}
                style={{
                  width: '100%',
                  padding: '10px',
                  borderRadius: '6px',
                  border: '1px solid #CCB66F',
                  background: '#fff',
                }}
              >
                <option value="">請選擇分店</option>
                {branches.map((branch) => (
                  <option key={branch.id} value={branch.id}>
                    {branch.name} ({branch.id})
                  </option>
                ))}
              </select>
              {selectedBranch && users.length === 0 && (
                <p style={{ margin: '8px 0 0', fontSize: '12px', color: '#8D6E63' }}>
                  此分店目前沒有與 {authInfo.coach_id} 建立 Students 關聯的學員。
                </p>
              )}
            </div>
          )}

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '8px',
            }}
          >
            <button
              onClick={() => setScanMode('body')}
              style={{
                padding: '12px 8px',
                borderRadius: '8px',
                border: scanMode === 'body' ? '2px solid #1976D2' : '1px solid #ddd',
                background: scanMode === 'body' ? '#E3F2FD' : '#fff',
                color: scanMode === 'body' ? '#0D47A1' : '#555',
                fontWeight: 'bold',
              }}
            >
              📊 身體組成
            </button>

            <button
              onClick={() => setScanMode('questionnaire')}
              style={{
                padding: '12px 8px',
                borderRadius: '8px',
                border:
                  scanMode === 'questionnaire'
                    ? '2px solid #6A1B9A'
                    : '1px solid #ddd',
                background: scanMode === 'questionnaire' ? '#F3E5F5' : '#fff',
                color: scanMode === 'questionnaire' ? '#4A148C' : '#555',
                fontWeight: 'bold',
              }}
            >
              📝 CoachOS 問卷
            </button>
          </div>

          {scanMode === 'body' && lastResult && (
            <div
              style={{
                background: '#FFFFFF',
                color: '#333333',
                border: '2px solid #4CAF50',
                padding: '20px',
                borderRadius: '12px',
                textAlign: 'left',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              }}
            >
              <h3
                style={{
                  margin: '0 0 15px 0',
                  color: '#2E7D32',
                  borderBottom: '2px solid #E8F5E9',
                  paddingBottom: '8px',
                  fontSize: '18px',
                }}
              >
                📊 雲端高階分析結果
              </h3>

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
                      {lastResult.source_system === 'muzili' ? 'MUZILI' : lastResult.device_type?.toUpperCase() || '--'}
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
                  background: '#E8F5E9',
                  padding: '12px',
                  borderRadius: '8px',
                  marginBottom: '15px',
                  border: '1px solid #C8E6C9',
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
                  <strong style={{ color: '#1B5E20' }}>
                    ⭐ CoachOS BCS
                  </strong>

                  <span
                    style={{
                      fontSize: '26px',
                      fontWeight: 'bold',
                      color: '#2E7D32',
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
                  background: '#F1F8E9',
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
                      color: '#2E7D32',
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
                    color: '#1976D2',
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
                    border: '1px solid #E3F2FD',
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
                    <strong style={{ color: '#1565C0' }}>
                      {segmental?.la ?? '--'}
                    </strong>
                  </div>

                  <div>
                    右上:
                    <br />
                    <strong style={{ color: '#1565C0' }}>
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
                    <strong style={{ color: '#1565C0' }}>
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
                    <strong style={{ color: '#1565C0' }}>
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
            </div>
          )}

          {scanMode === 'questionnaire' && questionnaireResult && (
            <div
              style={{
                background: '#FFFFFF',
                color: '#333',
                border: '2px solid #8E24AA',
                padding: '18px',
                borderRadius: '12px',
                textAlign: 'left',
                boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
              }}
            >
              <h3
                style={{
                  margin: '0 0 12px 0',
                  color: '#6A1B9A',
                  borderBottom: '2px solid #F3E5F5',
                  paddingBottom: '8px',
                }}
              >
                📝 今日 CoachOS 問卷辨識結果
              </h3>

              <div
                style={{
                  background: '#F7F3FA',
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
                <strong style={{ color: '#1565C0' }}>
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
            </div>
          )}

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '5px',
              textAlign: 'left',
            }}
          >
            <label style={{ fontWeight: 'bold' }}>👤 選擇學員:</label>

            <select
              value={selectedUser}
              onChange={handleSelectedUserChange}
              style={{
                padding: '12px',
                fontSize: '16px',
                borderRadius: '5px',
                border: questionnaireLockedUserId ? '2px solid #7B1FA2' : '1px solid #ccc',
              }}
            >
              {users.map((u) => (
                <option key={`${u.id}-${u.branch_id}-${u.coach_id}`} value={u.id}>
                  {u.name} ({String(u.id).substring(0, 5)}...)
                </option>
              ))}
            </select>

            {scanMode === 'questionnaire' && questionnaireLockedUserId && (
              <div
                style={{
                  marginTop: '6px',
                  padding: '8px 10px',
                  borderRadius: '6px',
                  background: '#F3E5F5',
                  color: '#6A1B9A',
                  fontSize: '12px',
                  fontWeight: 'bold',
                }}
              >
                🔒 本次未送出問卷已鎖定此學員；若切換學員，Page 1～4 尚未送出的照片會自動清空。
              </div>
            )}
          </div>

          {scanMode === 'body' && (
            <>
              <div
                style={{
                  background: '#E3F2FD',
                  border: '1px solid #90CAF9',
                  padding: '12px',
                  borderRadius: '8px',
                  fontSize: '13px',
                  color: '#0D47A1',
                  textAlign: 'left',
                }}
              >
                <strong>Muzili 身體組成資料匯入</strong>
                <br />
                請先輸入學員實際身高、年齡與性別，再依指定頁面上傳兩張 Muzili 圖片。
                <br />
                系統不會從人物圖、身體年齡或 FFMI 推測基本資料。
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '10px',
                  marginTop: '12px',
                }}
              >
                <div style={{ textAlign: 'left' }}>
                  <label style={{ fontWeight: 'bold' }}>身高 *</label>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      marginTop: '5px',
                    }}
                  >
                    <input
                      type="number"
                      inputMode="decimal"
                      min="100"
                      max="250"
                      step="0.1"
                      value={muziliHeightCm}
                      onChange={(event) => setMuziliHeightCm(event.target.value)}
                      placeholder="168.0"
                      style={{
                        width: '100%',
                        padding: '12px',
                        fontSize: '16px',
                        borderRadius: '6px',
                        border: '1px solid #ccc',
                        boxSizing: 'border-box',
                      }}
                    />
                    <span>cm</span>
                  </div>
                </div>

                <div style={{ textAlign: 'left' }}>
                  <label style={{ fontWeight: 'bold' }}>實際年齡 *</label>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      marginTop: '5px',
                    }}
                  >
                    <input
                      type="number"
                      inputMode="numeric"
                      min="1"
                      max="120"
                      step="1"
                      value={muziliAge}
                      onChange={(event) => setMuziliAge(event.target.value)}
                      placeholder="42"
                      style={{
                        width: '100%',
                        padding: '12px',
                        fontSize: '16px',
                        borderRadius: '6px',
                        border: '1px solid #ccc',
                        boxSizing: 'border-box',
                      }}
                    />
                    <span>歲</span>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: '10px', textAlign: 'left' }}>
                <label style={{ fontWeight: 'bold' }}>性別 *</label>
                <select
                  value={muziliGender}
                  onChange={(event) => setMuziliGender(event.target.value)}
                  style={{
                    width: '100%',
                    padding: '12px',
                    marginTop: '5px',
                    fontSize: '16px',
                    borderRadius: '6px',
                    border: '1px solid #ccc',
                    boxSizing: 'border-box',
                  }}
                >
                  <option value="">請選擇</option>
                  <option value="female">女性</option>
                  <option value="male">男性</option>
                </select>
              </div>

              <div
                style={{
                  border: muziliSlots.bodyData?.file
                    ? '2px solid #2196F3'
                    : '1px dashed #90A4AE',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  marginTop: '16px',
                }}
              >
                <div
                  style={{
                    padding: '10px',
                    background: '#E3F2FD',
                    textAlign: 'left',
                  }}
                >
                  <strong>① 身體數據 *</strong>
                  <div style={{ fontSize: '12px', marginTop: '5px', lineHeight: 1.6 }}>
                    請上傳 Muzili「身體數據」頁面。
                    <br />
                    畫面需包含：
                    <strong>體重、去脂體重、體脂率、內臟脂肪、基礎代謝、脂肪量</strong>
                  </div>
                </div>

                {muziliSlots.bodyData?.preview ? (
                  <img
                    src={muziliSlots.bodyData.preview}
                    alt="Muzili 身體數據"
                    style={{
                      width: '100%',
                      maxHeight: '320px',
                      objectFit: 'contain',
                      display: 'block',
                      background: '#FAFAFA',
                    }}
                  />
                ) : (
                  <div
                    style={{
                      height: '110px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#999',
                      fontSize: '13px',
                    }}
                  >
                    尚未選擇「身體數據」圖片
                  </div>
                )}

                <div style={{ padding: '10px' }}>
                  <label
                    htmlFor="muzili-body-data-input"
                    style={{
                      display: 'block',
                      padding: '12px',
                      background: '#2196F3',
                      color: 'white',
                      borderRadius: '7px',
                      fontWeight: 'bold',
                      textAlign: 'center',
                      cursor: 'pointer',
                    }}
                  >
                    {muziliSlots.bodyData?.file
                      ? '🔄 重新選擇「身體數據」圖片'
                      : '📁 選擇「身體數據」圖片'}
                  </label>
                  <input
                    id="muzili-body-data-input"
                    type="file"
                    accept="image/*"
                    onChange={(event) => handleMuziliImageUpload('bodyData', event)}
                    style={{ display: 'none' }}
                  />
                </div>
              </div>

              <div
                style={{
                  border: muziliSlots.segmentalMuscle?.file
                    ? '2px solid #00897B'
                    : '1px dashed #90A4AE',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  marginTop: '12px',
                }}
              >
                <div
                  style={{
                    padding: '10px',
                    background: '#E0F2F1',
                    textAlign: 'left',
                  }}
                >
                  <strong>② 節段肌肉分析 *</strong>
                  <div style={{ fontSize: '12px', marginTop: '5px', lineHeight: 1.6 }}>
                    請上傳 <strong>「節段分析 → 肌肉量」</strong>頁面。
                    <br />
                    畫面需完整顯示：<strong>左臂、右臂、左腿、右腿肌肉量</strong>。
                    <br />
                    <span style={{ color: '#C62828' }}>⚠️ 請勿上傳「脂肪量」頁面。</span>
                  </div>
                </div>

                {muziliSlots.segmentalMuscle?.preview ? (
                  <img
                    src={muziliSlots.segmentalMuscle.preview}
                    alt="Muzili 節段肌肉分析"
                    style={{
                      width: '100%',
                      maxHeight: '320px',
                      objectFit: 'contain',
                      display: 'block',
                      background: '#FAFAFA',
                    }}
                  />
                ) : (
                  <div
                    style={{
                      height: '110px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#999',
                      fontSize: '13px',
                    }}
                  >
                    尚未選擇「節段分析 → 肌肉量」圖片
                  </div>
                )}

                <div style={{ padding: '10px' }}>
                  <label
                    htmlFor="muzili-segmental-input"
                    style={{
                      display: 'block',
                      padding: '12px',
                      background: '#00897B',
                      color: 'white',
                      borderRadius: '7px',
                      fontWeight: 'bold',
                      textAlign: 'center',
                      cursor: 'pointer',
                    }}
                  >
                    {muziliSlots.segmentalMuscle?.file
                      ? '🔄 重新選擇「節段肌肉分析」圖片'
                      : '📁 選擇「節段肌肉分析」圖片'}
                  </label>
                  <input
                    id="muzili-segmental-input"
                    type="file"
                    accept="image/*"
                    onChange={(event) => handleMuziliImageUpload('segmentalMuscle', event)}
                    style={{ display: 'none' }}
                  />
                </div>
              </div>

              <p
                style={{
                  fontSize: '14px',
                  color: '#666',
                  margin: '8px 0',
                  fontWeight: 'bold',
                }}
              >
                {uploadStatus}
              </p>

              <button
                onClick={submitMuziliToApi}
                disabled={
                  !muziliHeightCm ||
                  !muziliAge ||
                  !muziliGender ||
                  !muziliSlots.bodyData?.file ||
                  !muziliSlots.segmentalMuscle?.file
                }
                style={{
                  padding: '15px',
                  background:
                    !muziliHeightCm ||
                    !muziliAge ||
                    !muziliGender ||
                    !muziliSlots.bodyData?.file ||
                    !muziliSlots.segmentalMuscle?.file
                      ? '#BDBDBD'
                      : '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  cursor:
                    !muziliHeightCm ||
                    !muziliAge ||
                    !muziliGender ||
                    !muziliSlots.bodyData?.file ||
                    !muziliSlots.segmentalMuscle?.file
                      ? 'not-allowed'
                      : 'pointer',
                }}
              >
                🚀 開始辨識並儲存
              </button>
            </>
          )}

          {scanMode === 'questionnaire' && (
            <>
              <div
                style={{
                  background: '#F3E5F5',
                  border: '1px solid #CE93D8',
                  padding: '12px',
                  borderRadius: '8px',
                  fontSize: '13px',
                  color: '#4A148C',
                  textAlign: 'left',
                }}
              >
                <strong>CoachOS 標準問卷拍照模式｜Page Slot 防呆</strong>
                <br />
                請把照片放進正確的 Page 1～Page 4。第一張照片拍下後，本次未送出問卷會鎖定目前學員；
                若切換學員，尚未送出的照片會自動清空。同一個 Page 重拍會直接取代舊照片，不會追加成另一頁。
                空白 Page 不會送出；可只補拍其中幾頁並與同一學員今日資料整合。
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: '10px',
                  margin: '12px 0',
                }}
              >
                {QUESTIONNAIRE_PAGE_NUMBERS.map((pageNumber) => {
                  const slot = questionnaireSlots[pageNumber]
                  const hasPhoto = Boolean(slot?.file)

                  return (
                    <div
                      key={pageNumber}
                      style={{
                        border: hasPhoto
                          ? '2px solid #8E24AA'
                          : '1px dashed #BDBDBD',
                        borderRadius: '10px',
                        overflow: 'hidden',
                        background: hasPhoto ? '#FCF7FF' : '#FAFAFA',
                      }}
                    >
                      <div
                        style={{
                          padding: '8px 10px',
                          background: hasPhoto ? '#F3E5F5' : '#F5F5F5',
                          color: hasPhoto ? '#6A1B9A' : '#666',
                          fontWeight: 'bold',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                        }}
                      >
                        <span>Page {pageNumber}</span>
                        <span style={{ fontSize: '11px' }}>
                          {hasPhoto ? '✅ 已拍攝' : '尚未拍攝'}
                        </span>
                      </div>

                      {slot?.preview ? (
                        <img
                          src={slot.preview}
                          alt={`CoachOS 問卷 Page ${pageNumber}`}
                          style={{
                            width: '100%',
                            height: '160px',
                            objectFit: 'cover',
                            display: 'block',
                          }}
                        />
                      ) : (
                        <div
                          style={{
                            height: '105px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: '#AAA',
                            fontSize: '13px',
                          }}
                        >
                          請拍攝第 {pageNumber} 頁
                        </div>
                      )}

                      <div
                        style={{
                          display: 'flex',
                          gap: '6px',
                          padding: '8px',
                        }}
                      >
                        <label
                          htmlFor={`questionnaire-camera-input-${pageNumber}`}
                          style={{
                            flex: 1,
                            padding: '9px 6px',
                            background: hasPhoto ? '#7B1FA2' : '#8E24AA',
                            color: 'white',
                            borderRadius: '6px',
                            fontSize: '13px',
                            fontWeight: 'bold',
                            textAlign: 'center',
                            cursor: 'pointer',
                          }}
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
                            style={{
                              border: 'none',
                              background: '#FFEBEE',
                              color: '#C62828',
                              borderRadius: '6px',
                              padding: '7px 9px',
                              fontWeight: 'bold',
                            }}
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
                <div
                  style={{
                    padding: '10px',
                    background: '#F7F3FA',
                    borderRadius: '8px',
                    fontSize: '13px',
                    color: '#5E356A',
                    textAlign: 'left',
                  }}
                >
                  <strong>本次待送出：</strong> Page{' '}
                  {questionnairePageNumbers.join('、')}（共 {questionnairePageCount} 頁）
                </div>
              )}

              <p
                style={{
                  fontSize: '14px',
                  color: '#666',
                  margin: '4px 0',
                  fontWeight: 'bold',
                }}
              >
                {questionnaireStatus}
              </p>

              <button
                onClick={submitQuestionnaire}
                disabled={questionnairePageCount === 0}
                style={{
                  padding: '15px',
                  background:
                    questionnairePageCount === 0 ? '#BDBDBD' : '#8E24AA',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  fontSize: '18px',
                  fontWeight: 'bold',
                  boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
                }}
              >
                📝 辨識並整合 ({questionnairePageCount} 頁)
              </button>

              {questionnairePageCount > 0 && (
                <button
                  type="button"
                  onClick={() =>
                    clearQuestionnaireDraft('已清空本次未送出的問卷照片，學員鎖定已解除')
                  }
                  style={{
                    padding: '11px',
                    background: '#FFFFFF',
                    color: '#6A1B9A',
                    border: '1px solid #CE93D8',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: 'bold',
                  }}
                >
                  🗑️ 清空 Page 1～8 未送出照片
                </button>
              )}
            </>
          )}

        </div>
      )}
    </div>
  )
}

export default App
