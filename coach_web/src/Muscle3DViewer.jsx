import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const ASSET_BASE = '/resources/muscle3d/'
const DRACO_DECODER = `${ASSET_BASE}draco/`

const BASE_COLOR = new THREE.Color('#C98E79')
const HIGHLIGHT_COLOR = new THREE.Color('#D85A57')
const SKELETON_COLOR = new THREE.Color('#E8DFCC')
const BACKGROUND_COLOR = new THREE.Color('#F5F7F8')

// Business Muscle No. without a valid 3D target. These are intentionally ignored.
export const UNSUPPORTED_MUSCLE_CODES = new Set([5, 6, 8, 9, 10, 11, 20, 21, 46, 65, 126, 157])


const MUSCLE_GROUP_LABELS = {
  'Neck Muscles': '頸部肌群',
  'Shoulder Muscles': '肩部肌群',
  'Front Arm Muscles': '上臂前側肌群',
  'Back Arm Muscles': '上臂後側肌群',
  'Chest Muscles': '胸部肌群',
  'Back Muscles': '背部肌群',
  'Abdominal Muscle': '核心／腹部肌群',
  'Gluteal Muscles': '臀部肌群',
  'Quadriceps Muscles': '股四頭肌群',
  'Hamstrings and Adductors': '腿後／內收肌群',
  'Knee Extensors': '膝伸肌群',
  'Knee Flexors': '膝屈肌群',
  'Front Calf Muscles': '小腿前側肌群',
  'Back Calf Muscles': '小腿後側肌群',
}

const VIEW_LABELS = {
  front: '正面',
  right: '右側',
  left: '左側',
  back: '背面',
}

const mappingPromise = { current: null }

function normalizeRiskCodes(values = []) {
  return [...new Set(
    (values || [])
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value > 0)
      .map(Math.round),
  )]
}

async function loadMapping() {
  if (!mappingPromise.current) {
    mappingPromise.current = fetch(`${ASSET_BASE}muscle_3d_mapping.json`, { cache: 'force-cache' })
      .then((response) => {
        if (!response.ok) throw new Error(`muscle_3d_mapping.json HTTP ${response.status}`)
        return response.json()
      })
  }
  return mappingPromise.current
}

function cloneForLeft(root) {
  const left = root.clone(true)
  left.scale.x *= -1
  left.traverse((obj) => {
    if (obj.isMesh && obj.material) {
      obj.material = Array.isArray(obj.material)
        ? obj.material.map((material) => material.clone())
        : obj.material.clone()
    }
  })
  return left
}

function normalizeMuscleMesh(mesh, side, structureIndex, muscleMeshes) {
  const structureId = mesh.userData?.structure_id || mesh.parent?.userData?.structure_id || 'unknown'
  mesh.userData.structure_id = structureId
  mesh.userData.side = side
  mesh.userData.original_name = mesh.name || ''
  mesh.userData.stable_key = `${structureId}|${side}`
  const previousMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
  previousMaterials.filter(Boolean).forEach((material) => material.dispose?.())
  mesh.material = new THREE.MeshStandardMaterial({
    color: BASE_COLOR.clone(),
    roughness: 0.84,
    metalness: 0,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 1,
  })
  muscleMeshes.push(mesh)
  if (!structureIndex.has(mesh.userData.stable_key)) structureIndex.set(mesh.userData.stable_key, [])
  structureIndex.get(mesh.userData.stable_key).push(mesh)
}

function normalizeSkeletonMesh(mesh, skeletonMeshes) {
  const previousMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
  previousMaterials.filter(Boolean).forEach((material) => material.dispose?.())
  mesh.material = new THREE.MeshStandardMaterial({
    color: SKELETON_COLOR.clone(),
    roughness: 0.58,
    metalness: 0.02,
    transparent: true,
    opacity: 0.52,
    depthWrite: false,
    side: THREE.FrontSide,
  })
  mesh.renderOrder = 0
  skeletonMeshes.push(mesh)
}

function setRendererDefaults(renderer) {
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.88
}

async function buildScene({ canvas, width, height, interactive = false }) {
  const scene = new THREE.Scene()
  scene.background = BACKGROUND_COLOR.clone()

  const camera = new THREE.PerspectiveCamera(30, Math.max(width, 1) / Math.max(height, 1), 0.01, 1000)
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: !interactive,
    powerPreference: 'high-performance',
  })
  setRendererDefaults(renderer)
  renderer.setPixelRatio(interactive ? Math.min(window.devicePixelRatio || 1, 2) : 1)
  renderer.setSize(width, height, false)

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c0c5, 1.42))
  const key = new THREE.DirectionalLight(0xffffff, 1.62)
  key.position.set(4, 7, 7)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0xc9d1d8, 0.72)
  rim.position.set(-5, 3, -6)
  scene.add(rim)

  const draco = new DRACOLoader()
  draco.setDecoderPath(DRACO_DECODER)
  const loader = new GLTFLoader()
  loader.setDRACOLoader(draco)

  const [muscleGltf, skeletonGltf] = await Promise.all([
    loader.loadAsync(`${ASSET_BASE}muscular_lite.glb`),
    loader.loadAsync(`${ASSET_BASE}skeletal_lite.glb`),
  ])
  draco.dispose()

  const structureIndex = new Map()
  const muscleMeshes = []
  const skeletonMeshes = []
  const group = new THREE.Group()
  scene.add(group)

  const muscleRight = muscleGltf.scene
  const muscleLeft = cloneForLeft(muscleRight)
  muscleRight.traverse((obj) => { if (obj.isMesh) normalizeMuscleMesh(obj, 'right', structureIndex, muscleMeshes) })
  muscleLeft.traverse((obj) => { if (obj.isMesh) normalizeMuscleMesh(obj, 'left', structureIndex, muscleMeshes) })

  const skeletonRight = skeletonGltf.scene
  const skeletonLeft = cloneForLeft(skeletonRight)
  skeletonRight.traverse((obj) => { if (obj.isMesh) normalizeSkeletonMesh(obj, skeletonMeshes) })
  skeletonLeft.traverse((obj) => { if (obj.isMesh) normalizeSkeletonMesh(obj, skeletonMeshes) })

  group.add(skeletonRight)
  group.add(skeletonLeft)
  group.add(muscleRight)
  group.add(muscleLeft)

  const box = new THREE.Box3().setFromObject(group)
  const center = box.getCenter(new THREE.Vector3())
  group.position.sub(center)

  const finalBox = new THREE.Box3().setFromObject(group)
  const finalSize = finalBox.getSize(new THREE.Vector3())
  const verticalFov = THREE.MathUtils.degToRad(camera.fov)
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect)
  const fitHeightDistance = finalSize.y / (2 * Math.tan(verticalFov / 2))
  const fitWidthDistance = finalSize.x / (2 * Math.tan(horizontalFov / 2))
  const fitPadding = interactive ? 1.16 : 1.08
  const dist = Math.max(fitHeightDistance, fitWidthDistance) * fitPadding
  const targetY = finalSize.y * 0.015

  camera.near = Math.max(dist / 500, 0.01)
  camera.far = dist * 20
  camera.updateProjectionMatrix()

  let controls = null
  if (interactive) {
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.enablePan = false
    controls.minDistance = dist * 0.72
    controls.maxDistance = dist * 1.65
    controls.target.set(0, targetY, 0)
  }

  const setView = (view = 'front') => {
    // Source right-side GLB occupies negative X. Therefore the anatomical
    // right-side view is camera -X; the original mapper had these reversed.
    if (view === 'front') camera.position.set(0, targetY, dist)
    else if (view === 'back') camera.position.set(0, targetY, -dist)
    else if (view === 'right') camera.position.set(-dist, targetY, 0)
    else if (view === 'left') camera.position.set(dist, targetY, 0)
    else camera.position.set(0, targetY, dist)

    camera.up.set(0, 1, 0)
    camera.lookAt(0, targetY, 0)
    if (controls) {
      controls.target.set(0, targetY, 0)
      controls.update()
    }
    renderer.render(scene, camera)
  }

  setView('front')

  const dispose = () => {
    if (controls) controls.dispose()
    scene.traverse((obj) => {
      if (!obj.isMesh) return
      if (obj.geometry) obj.geometry.dispose()
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material]
      materials.filter(Boolean).forEach((material) => material.dispose())
    })
    renderer.dispose()
    renderer.forceContextLoss?.()
  }

  return {
    scene,
    camera,
    renderer,
    controls,
    structureIndex,
    muscleMeshes,
    skeletonMeshes,
    setView,
    dispose,
    dist,
    targetY,
  }
}

function resetMaterials(bundle) {
  bundle.muscleMeshes.forEach((mesh) => {
    mesh.material.color.copy(BASE_COLOR)
    mesh.material.opacity = 1
    mesh.visible = true
  })
  bundle.skeletonMeshes.forEach((mesh) => {
    mesh.material.color.copy(SKELETON_COLOR)
    mesh.material.opacity = 0.52
    mesh.visible = true
  })
}

export async function applyRiskHighlights(bundle, riskMuscles = []) {
  const mapping = await loadMapping()
  const codes = normalizeRiskCodes(riskMuscles)
  resetMaterials(bundle)

  const highlightedMeshes = new Set()
  const groupNames = new Set()
  const skippedCodes = []

  for (const code of codes) {
    if (UNSUPPORTED_MUSCLE_CODES.has(code)) {
      skippedCodes.push(code)
      continue
    }

    const entry = mapping?.mappings?.[String(code)]
    if (!entry?.structures?.length) {
      skippedCodes.push(code)
      continue
    }

    let codeMatched = false
    for (const target of entry.structures) {
      const key = `${target.structure_id}|${target.side}`
      const meshes = bundle.structureIndex.get(key) || []
      if (!meshes.length) continue

      const wantedNames = new Set(target.mesh_names || [])
      const targetMeshes = wantedNames.size
        ? meshes.filter((mesh) => wantedNames.has(mesh.userData.original_name))
        : meshes

      targetMeshes.forEach((mesh) => {
        mesh.material.color.copy(HIGHLIGHT_COLOR)
        mesh.material.opacity = 1
        highlightedMeshes.add(mesh)
        codeMatched = true
      })
    }

    if (codeMatched) {
      ;(entry.groups || []).forEach((group) => groupNames.add(MUSCLE_GROUP_LABELS[group] || group))
    } else {
      skippedCodes.push(code)
    }
  }

  const skippedSet = new Set(skippedCodes)
  bundle.renderer.render(bundle.scene, bundle.camera)
  return {
    codes,
    displayCodes: codes.filter((code) => !skippedSet.has(code)),
    highlightedCount: highlightedMeshes.size,
    groups: [...groupNames],
    skippedCodes,
  }
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()))
}

function getHostRenderSize(host) {
  const rect = host.getBoundingClientRect()
  return {
    // The renderer aspect ratio must match the CSS box exactly. A 320px
    // minimum made narrow Android layouts render wide and then squeeze the
    // canvas horizontally when CSS fitted it back into the host.
    width: Math.max(1, Math.round(rect.width)),
    height: Math.max(1, Math.round(rect.height)),
  }
}

function trimmedCanvasDataUrl(sourceCanvas) {
  const width = sourceCanvas.width
  const height = sourceCanvas.height
  const scratch = document.createElement('canvas')
  scratch.width = width
  scratch.height = height

  const ctx = scratch.getContext('2d', { willReadFrequently: true })
  if (!ctx) return sourceCanvas.toDataURL('image/png')

  ctx.drawImage(sourceCanvas, 0, 0)
  const pixels = ctx.getImageData(0, 0, width, height).data
  const cornerOffsets = [
    0,
    (width - 1) * 4,
    ((height - 1) * width) * 4,
    ((height * width) - 1) * 4,
  ]
  const bg = cornerOffsets.reduce(
    (sum, index) => ({
      r: sum.r + pixels[index],
      g: sum.g + pixels[index + 1],
      b: sum.b + pixels[index + 2],
    }),
    { r: 0, g: 0, b: 0 },
  )
  bg.r /= cornerOffsets.length
  bg.g /= cornerOffsets.length
  bg.b /= cornerOffsets.length
  const tolerance = 24

  let minX = width
  let minY = height
  let maxX = -1
  let maxY = -1

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      const index = (y * width + x) * 4
      const r = pixels[index]
      const g = pixels[index + 1]
      const b = pixels[index + 2]
      const differentFromBackground =
        Math.abs(r - bg.r) + Math.abs(g - bg.g) + Math.abs(b - bg.b) > tolerance

      if (!differentFromBackground) continue
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }

  if (maxX < minX || maxY < minY) {
    return sourceCanvas.toDataURL('image/png')
  }

  const contentWidth = maxX - minX + 1
  const contentHeight = maxY - minY + 1
  const padX = Math.round(contentWidth * 0.08)
  const padY = Math.round(contentHeight * 0.035)
  const sx = Math.max(0, minX - padX)
  const sy = Math.max(0, minY - padY)
  const sw = Math.min(width - sx, contentWidth + padX * 2)
  const sh = Math.min(height - sy, contentHeight + padY * 2)

  const output = document.createElement('canvas')
  output.width = sw
  output.height = sh
  const outCtx = output.getContext('2d')
  if (!outCtx) return sourceCanvas.toDataURL('image/png')

  outCtx.fillStyle = '#F5F7F8'
  outCtx.fillRect(0, 0, sw, sh)
  outCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, sw, sh)
  return output.toDataURL('image/png')
}

export async function renderMuscleReportSnapshots(riskMuscles = [], options = {}) {
  // Tall report canvas follows the natural full-body proportion used in the reference view.
  const width = options.width || 900
  const height = options.height || 1800
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const bundle = await buildScene({ canvas, width, height, interactive: false })
  try {
    const meta = await applyRiskHighlights(bundle, riskMuscles)
    const images = {}

    for (const view of ['front', 'back']) {
      bundle.setView(view)
      bundle.renderer.render(bundle.scene, bundle.camera)
      await nextFrame()
      bundle.renderer.render(bundle.scene, bundle.camera)
      images[view] = trimmedCanvasDataUrl(bundle.renderer.domElement)
    }

    return { ...images, ...meta }
  } finally {
    bundle.dispose()
  }
}

export default function Muscle3DViewer({ riskMuscles = [], sourceDate = '' }) {
  const hostRef = useRef(null)
  const canvasRef = useRef(null)
  const bundleRef = useRef(null)
  const frameRef = useRef(null)
  const codesRef = useRef([])
  const [activeView, setActiveView] = useState('front')
  const [status, setStatus] = useState('loading')
  const [meta, setMeta] = useState({ groups: [], skippedCodes: [], highlightedCount: 0 })

  const codes = useMemo(() => normalizeRiskCodes(riskMuscles), [riskMuscles])
  const riskKey = useMemo(() => codes.join(','), [codes])
  codesRef.current = codes

  useEffect(() => {
    let cancelled = false
    let resizeObserver = null

    const boot = async () => {
      setStatus('loading')
      try {
        const host = hostRef.current
        const canvas = canvasRef.current
        if (!host || !canvas) return

        const { width, height } = getHostRenderSize(host)
        const bundle = await buildScene({
          canvas,
          width,
          height,
          interactive: true,
        })
        if (cancelled) {
          bundle.dispose()
          return
        }

        bundleRef.current = bundle
        bundle.setView('front')
        const applied = await applyRiskHighlights(bundle, codesRef.current)
        if (!cancelled) {
          setMeta(applied)
          setStatus('ready')
        }

        const animate = () => {
          if (cancelled || !bundleRef.current) return
          bundle.controls?.update()
          bundle.renderer.render(bundle.scene, bundle.camera)
          frameRef.current = requestAnimationFrame(animate)
        }
        animate()

        resizeObserver = new ResizeObserver(() => {
          const current = bundleRef.current
          const nextHost = hostRef.current
          if (!current || !nextHost) return
          const { width, height } = getHostRenderSize(nextHost)
          current.camera.aspect = width / height
          current.camera.updateProjectionMatrix()
          current.renderer.setSize(width, height, false)
          current.renderer.render(current.scene, current.camera)
        })
        resizeObserver.observe(host)
      } catch (error) {
        console.error('3D muscle viewer load failed:', error)
        if (!cancelled) setStatus('error')
      }
    }

    boot()

    return () => {
      cancelled = true
      resizeObserver?.disconnect()
      if (frameRef.current) cancelAnimationFrame(frameRef.current)
      frameRef.current = null
      if (bundleRef.current) bundleRef.current.dispose()
      bundleRef.current = null
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      if (!bundleRef.current) return
      try {
        const applied = await applyRiskHighlights(bundleRef.current, codes)
        if (!cancelled) setMeta(applied)
      } catch (error) {
        console.error('3D muscle highlight refresh failed:', error)
      }
    }
    refresh()
    return () => { cancelled = true }
  }, [riskKey])

  const chooseView = (view) => {
    setActiveView(view)
    bundleRef.current?.setView(view)
  }

  return (
    <section className="risk-muscle-block risk-muscle-3d-block">
      <div className="risk-muscle-copy">
        <div>
          <div className="eyebrow">RISK MUSCLE MAP</div>
          <h3>體測待強化肌群</h3>
          <p>{sourceDate ? `資料日期 ${sourceDate}` : '尚無體測日期'}</p>
        </div>
      </div>

      <div className="muscle-3d-view-buttons" role="group" aria-label="肌肉圖視角">
        {Object.entries(VIEW_LABELS).map(([key, label]) => (
          <button
            type="button"
            key={key}
            className={activeView === key ? 'active' : ''}
            onClick={() => chooseView(key)}
          >
            {label}
          </button>
        ))}
      </div>

      <div ref={hostRef} className="muscle-3d-host">
        <canvas ref={canvasRef} className="muscle-3d-canvas" />
        {status === 'loading' && <div className="muscle-3d-overlay">正在載入 3D 肌肉模型...</div>}
        {status === 'error' && <div className="muscle-3d-overlay error">3D 肌肉模型載入失敗，請確認 muscle3d assets 與 Three.js。</div>}
      </div>

      {!!meta.groups?.length && (
        <div className="risk-group-list">
          {meta.groups.map((group) => <span key={group}>{group}</span>)}
        </div>
      )}
      {!codes.length && <div className="risk-empty">最新體測未標記 risk_muscles。</div>}

    </section>
  )
}
