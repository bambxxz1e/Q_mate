<template>
  <div class="result-wrap">
    <h1 class="title">모의 면접 결과</h1>

    <section class="top">
      <!-- 좌측: 증명사진 + 기본 정보 -->
      <div class="left">
        <div class="photo-box">
          <img v-if="photo" :src="photo" alt="증명사진" class="photo" />
          <div v-else class="photo-empty">사진이 없습니다</div>
        </div>

        <div class="info">
          <div class="row">
            <span class="label">이름 :</span>
            <span class="name">{{ name || '이름 미등록' }}</span>
          </div>

          <button class="badge">{{ position }}</button>

          <div class="intro">
            <h3>한줄 자기소개</h3>
            <p>{{ intro }}</p>
          </div>
        </div>
      </div>

      <!-- 우측: 원형 점수 -->
      <div class="right">
        <div class="circle">
          <svg :width="size" :height="size" class="ring">
            <circle
              class="bg"
              :cx="center"
              :cy="center"
              :r="radius"
              fill="transparent"
              :stroke-width="stroke"
            />
            <circle
              class="fg"
              :cx="center"
              :cy="center"
              :r="radius"
              fill="transparent"
              :stroke-width="stroke"
              :stroke-dasharray="circumference"
              :stroke-dashoffset="dashoffset"
              stroke-linecap="round"
              :style="{ transition: `stroke-dashoffset ${duration}ms ease` }"
            />
          </svg>
          <div class="score">
            <div class="percent">{{ Math.round(animatedScore) }}%</div>
            <div class="caption">모의 면접 점수</div>
          </div>
        </div>
      </div>
    </section>

    <!-- 하단: 이해를 돕는 이미지 + 설명 -->
    <section class="bottom">
      <h3 class="sub-title">면접 AI가 캐치한 내 태도</h3>

      <div class="gallery">
        <img :src="helperImg1" alt="예시 이미지 1" class="helper" />
        <img :src="helperImg2" alt="예시 이미지 2" class="helper" />

        <div class="notes">
          <p>
            <strong>{{ feedbackTitle }}</strong>이 감지되었어요!
          </p>
          <p class="muted">
            {{ feedbackMessage }}
          </p>
          <p class="muted small">(* 이해를 돕기 위한 예시 이미지 입니다)</p>
        </div>
      </div>
    </section>

    <!-- 저장 버튼 영역 -->
    <section class="actions">
      <button class="save-btn" @click="handleSaveClick" :disabled="isSaving">
        <span v-if="isSaving">저장 중...</span>
        <span v-else>결과 저장하기</span>
      </button>
      <p class="save-hint">※ 저장된 결과는 향후 마이페이지에서 조회할 수 있어요.</p>
    </section>

    <div class="next-wrap">
      <button class="next-btn" @click="goNext">나가기 →</button>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, nextTick } from 'vue'
import { supabase, getCurrentUser } from '../supabaseClient'
import { useRoute, useRouter } from 'vue-router'
import './Result.css'

const STORAGE_KEY = 'interviewForm'
const router = useRouter()
const route = useRoute()

// ===== 화면 상태 =====
const photo = ref(null)
const position = ref(route.query.position || '취업')
const feedbackTitle = ref(route.query.feedback_title || 'AI 종합 피드백')
const feedbackMessage = ref(route.query.feedback_message || '')
const intro = ref('안녕하세요! 저는 어제보다 더 나은 내일을 만드는 FE 개발자 입니다!')
const name = ref('')

const isSaving = ref(false) // 저장 중 상태

// ===== 게이지/숫자 애니메이션 =====
const score = ref(route.query.avgScore !== undefined ? Number(route.query.avgScore) : 50)
const animatedScore = ref(0)
const duration = 1000

const size = 220
const stroke = 12
const center = computed(() => size / 2)
const radius = computed(() => center.value - stroke / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)
const dashoffset = computed(() => {
  const ratio = Math.max(0, Math.min(100, animatedScore.value)) / 100
  return circumference.value * (1 - ratio)
})

function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3)
}

async function startAnimation() {
  await nextTick()
  const start = performance.now()

  function frame(now) {
    const raw = Math.min((now - start) / duration, 1)
    const eased = easeOutCubic(raw)
    animatedScore.value = score.value * eased
    if (raw < 1) requestAnimationFrame(frame)
  }
  requestAnimationFrame(frame)
}

// ===== DB에 결과 저장 =====
async function saveResultToSupabase() {
  const user = await getCurrentUser()
  if (!user) {
    alert('로그인 정보를 찾을 수 없습니다!')
    return
  }

  const payload = {
    user_id: user.id,
    form_id: null, // 나중에 interview_forms랑 연결하면 여기 넣기
    score: score.value,
    position: position.value,
    intro: intro.value,
    photo_url: photo.value,
    feedback_title: feedbackTitle.value,
    feedback_message: feedbackMessage.value
  }

  console.log('[Result] 저장 요청:', payload)

  const { data, error } = await supabase
    .from('interview_results')
    .insert(payload)
    .select()
    .single()

  if (error) {
    console.error('[Result] 저장 실패:', error)
    alert('저장 중 오류가 발생했습니다.')
    return
  }

  console.log('[Result] 저장 성공:', data)
  alert('면접 결과가 저장되었습니다!')
}

// 버튼 클릭 핸들러 (중복 클릭 방지)
async function handleSaveClick() {
  if (isSaving.value) return
  isSaving.value = true
  try {
    await saveResultToSupabase()
  } finally {
    isSaving.value = false
  }
}

// ===== Supabase + 로컬 데이터 로드 =====
async function loadResultData() {
  try {
    // 1) 로그인 유저 확인
    const user = await getCurrentUser()
    console.log('[Result] getCurrentUser:', user)

    if (!user) {
      alert('로그인 정보가 없습니다. 다시 로그인 후 이용해 주세요.')
      return
    }

    // 2) users 테이블에서 프로필 정보 불러오기
    const { data: profile, error: profileError } = await supabase
      .from('users')
      .select('name, intro, bio, photo_url')
      .eq('id', user.id)
      .single()

    if (profileError) {
      console.error('[Result] 프로필 조회 실패:', profileError)
    } else if (profile) {
      if (profile.name) name.value = profile.name
      if (profile.intro || profile.bio) {
        intro.value = (profile.intro || profile.bio).trim()
      }
      if (profile.photo_url) {
        photo.value = profile.photo_url
      }
    }

    // 3) 최근 인터뷰 폼(interview_forms)에서 추가 정보 불러오기 (있으면)
    try {
      const { data: form, error: formError } = await supabase
        .from('interview_forms')
        .select('position, photo')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .single()

      if (formError) {
        console.log('[Result] 인터뷰 폼 조회 실패 또는 없음:', formError.message)
      } 
      if (form) {
        if (!position.value && form.position) position.value = form.position
        if (form.photo) photo.value = form.photo
      }
    } catch (formErr) {
      console.error('[Result] 인터뷰 폼 조회 중 예외:', formErr)
    }

    // 4) Supabase에 아무것도 없을 경우를 대비한 localStorage fallback
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const parsed = JSON.parse(raw)
        if (!photo.value && parsed?.photo) photo.value = parsed.photo
        if (!position.value && parsed?.position) position.value = parsed.position
        const first = Array.isArray(parsed?.questions) ? parsed.questions[0] : null
        if (first?.content?.trim() && intro.value === '') {
          intro.value = first.content.trim()
        }
      }
    } catch (lsErr) {
      console.error('[Result] localStorage 로드 실패:', lsErr)
    }
  } catch (e) {
    console.error('[Result] 전체 로드 실패:', e)
  }
}

// 마운트 시 데이터 로드 + 점수 애니메이션만
onMounted(async () => {
  await loadResultData()
  startAnimation()
})
// 저장은 버튼에서만 실행 (자동 저장 X)
const helperImg1 = ref('https://picsum.photos/seed/hand-talk/480/320')
const helperImg2 = ref('https://picsum.photos/seed/hand-pose/480/320')

async function goNext() {
  router.push({
    path: '/myPage',
  })
}
</script>

<style src="./Result.css" scoped></style>
