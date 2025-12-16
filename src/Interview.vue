<template>
  <div class="interview-container">
    <div class="interview-page">
      <!-- Header strip -->
      <header class="topbar">
        <h1 class="brand">면접 연습하기</h1>
        <div class="status-pill" :class="{ on: micOn }">
          <span class="dot" />
          <span>{{ micOn ? '마이크 ON' : '마이크 OFF' }}</span>
        </div>
      </header>

      <main class="stage">
        <!-- Left: Interviewers window -->
        <section class="panel interviewers">
          <div class="panel-head">
            <span class="win-dot" />
            <span class="win-dot" />
            <span class="win-dot" />
          </div>
          <div class="panel-body">
            <img v-if="interviewerImageSrc" :src="interviewerImageSrc" alt="면접관 화면" />
            <div v-else class="img-placeholder">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="56" height="56">
                <path
                  d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-3.33 0-10 1.67-10 5v1h20v-1c0-3.33-6.67-5-10-5z" />
              </svg>
              <p>면접관 영상 자리 (이미지로 교체)</p>
            </div>
          </div>
        </section>

        <!-- Right: My camera window -->
        <section class="panel mycam">
          <div class="panel-head">
            <span class="win-dot" />
            <span class="win-dot" />
            <span class="win-dot" />
          </div>

          <div class="panel-body cam-body">
            <!-- ✅ 실제 카메라 스트림 -->
            <div v-if="cameraOn" class="cam-stream">
              <video
                ref="videoRef"
                autoplay
                playsinline
                class="cam-video"
              ></video>

              <button type="button" class="cam-cta small" @click="stopCamera">
                카메라 끄기
              </button>
            </div>

            <!-- 아직 카메라 안 켰을 때 CTA -->
            <button
              v-else
              type="button"
              class="cam-cta"
              @click="startCamera"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="44" height="44">
                <path
                  d="M20 5h-3.17L15 3H9L7.17 5H4c-1.1 0-2 .9-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7c0-1.1-.9-2-2-2zm0 14H4V7h4.05l1.83-2h4.24l1.83 2H20v12z" />
              </svg>
              내 카메라 화면 켜기
            </button>
          </div>
        </section>

        <!-- Tip card -->
        <section class="tip-session">
          <aside class="tip">
            <div class="tip-head">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18">
                <path
                  d="M9 21h6v-1H9v1zm3-20C6.48 1 2 5.48 2 11c0 3.53 1.84 6.62 4.6 8.4V21h10.8v-1.6C20.16 17.62 22 14.53 22 11 22 5.48 17.52 1 12 1zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" />
              </svg>
              <span>TIP</span>
            </div>
            <p>말을 더듬지 않고, 또박또박 말해요!</p>
          </aside>

          <!-- AI 조언 채팅 영역 -->
          <div class="chat-advice" ref="chatAdviceContainer"></div>
        </section>
      </main>

      <!-- Bottom: AI Question Bar -->
      <footer class="qa-bar">
        <span class="badge">면접관</span>

        <div class="question">
          <span class="label">AI 질문</span>
          <div :class="['qtext', loading ? 'loading' : (questionText.includes('오류') ? 'error' : 'normal')]">
            <template v-if="loading">
              <span class="skeleton" />
              <span class="skeleton" />
              <span class="skeleton short" />
            </template>
            <span v-else>{{ questionText || '질문을 받아오는 중입니다…' }}</span>
          </div>
        </div>

        <button class="mic" @click="handleMicClick" :aria-pressed="micOn">
          <div class="pulse" :class="{ active: micOn }" />
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
            <path
              d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 14 0h-2zM11 19h2v3h-2z" />
          </svg>
        </button>

        <button class="refresh" @click="refreshQuestion">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24">
            <path
              d="M12 6V3L8 7l4 4V8c2.76 0 5 2.24 5 5a5 5 0 0 1-8.66 3.54l-1.42 1.42A7 7 0 0 0 19 13c0-3.87-3.13-7-7-7z" />
          </svg>
          새 질문
        </button>
      </footer>

      <div class="next-wrap">
        <button class="next-btn" @click="goNext">면접 종료하기 →</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watchEffect, onBeforeUnmount, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import InterviewerImg from './img/Interview.png'
import axios from 'axios'

const route = useRoute()
const router = useRouter()
const position = ref(route.query.position || '일반')

// ✅ props
const props = defineProps<{
  cameraOn?: boolean
}>()

// ✅ emit 정의
const emit = defineEmits<{
  (e: 'toggle-mic', value: boolean): void
  (e: 'start-camera'): void
  (e: 'stop-camera'): void
  (e: 'refresh-question'): void
}>()

// ✅ 면접관 이미지
const interviewerImageSrc = InterviewerImg

// ✅ 마이크 상태
const micOn = ref(false)

// ✅ 카메라 관련 상태
const cameraOn = ref(!!props.cameraOn)
watchEffect(() => {
  cameraOn.value = !!props.cameraOn || cameraOn.value
})

// ✅ 실제 video 엘리먼트 & 스트림
const videoRef = ref<HTMLVideoElement | null>(null)
const stream = ref<MediaStream | null>(null)

// 마이크 토글
function toggleMic() {
  micOn.value = !micOn.value
  emit('toggle-mic', micOn.value)
}

// 오디오 관련 상태
const isRecording = ref(false)
const recordedBlob = ref<Blob | null>(null)

let mediaRecorder: MediaRecorder | null = null

async function startRecording() {
  if (isRecording.value) return

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)
    const chunks: BlobPart[] = []

    mediaRecorder.ondataavailable = (e) => chunks.push(e.data)
    mediaRecorder.onstop = () => {
      recordedBlob.value = new Blob(chunks, { type: 'audio/webm' })
      sendToAI(recordedBlob.value)
    }

    mediaRecorder.start()
    isRecording.value = true

    // 답변 시간 60초 제한
    setTimeout(() => stopRecording(), 60000)
  } catch (err) {
    console.error('마이크를 켤 수 없습니다:', err)
    alert('마이크 권한을 허용했는지 확인해주세요.')
  }
}

function stopRecording() {
  if (!mediaRecorder) return
  mediaRecorder.stop()
  mediaRecorder = null
  isRecording.value = false
}

function handleMicClick() {
  toggleMic()
  if (!isRecording.value) {
    startRecording()
  } else {
    stopRecording()
  }
}

// ✅ 배열
const scores = ref<number[]>([])
const adviceList = ref<string[]>([])

// ✅ 질문 카운트
const questionCount = ref(0)
const maxQuestions = 5

const chatAdviceContainer = ref<HTMLElement | null>(null)

async function sendToAI(blob: Blob) {
  if (!blob) return
  const formData = new FormData()
  formData.append('file', blob)
  formData.append('question', questionText.value)

  try {
    const res = await axios.post('http://localhost:8000/api/evaluate-answer', formData)
    const newScore = res.data.score
    const advice = res.data.advice

    adviceList.value.push(advice)
    scores.value.push(newScore)
    questionCount.value++

    // ✅ AI 조언 채팅처럼 쌓기
    if (chatAdviceContainer.value && advice) {
      const msg = document.createElement('div')
      msg.className = 'advice-msg'
      msg.textContent = advice
      chatAdviceContainer.value.appendChild(msg)
      // 스크롤 항상 맨 아래
      chatAdviceContainer.value.scrollTop = chatAdviceContainer.value.scrollHeight
    }

    if (questionCount.value >= maxQuestions) {
      // ✅ 질문 모두 끝나면 평균 계산 후 결과로 이동
      const avgScore = scores.value.reduce((a, b) => a + b, 0) / scores.value.length
      const summary = await requestSummary()
      
      router.push({
        path: '/result',
        query: {
          avgScore,
          position: position.value,
          feedback_title: summary.feedback_title,
          feedback_message: summary.feedback_message
        }
      })
    } else {
      // ✅ 다음 질문 자동 요청
      refreshQuestion()
    }
  } catch (err) {
    console.error(err)
    alert('답변 평가 중 오류가 발생했습니다.')
  }
}

// 피드백 요약
async function requestSummary() {
  const res = await axios.post(
    'http://localhost:8000/api/interview-summary',
    {
      advices: adviceList.value
    }
  )

  return res.data
}

// 카메라 시작
async function startCamera() {
  try {
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    })

    stream.value = mediaStream
    cameraOn.value = true

    await nextTick()
    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream
    }

    emit('start-camera')
  } catch (err) {
    console.error('카메라를 켤 수 없습니다:', err)
    alert('카메라 권한을 허용했는지, 브라우저/https 환경인지 확인해주세요.')
  }
}

// 카메라 정지
function stopCamera() {
  if (stream.value) {
    stream.value.getTracks().forEach((track) => track.stop())
    stream.value = null
  }
  cameraOn.value = false
  emit('stop-camera')
}

// 컴포넌트가 사라질 때 카메라 반드시 정지
onBeforeUnmount(() => {
  stopCamera()
})

async function goNext() {
  // scores 배열에서 평균 계산
  const avgScore =
    scores.value.length > 0
      ? Math.round(scores.value.reduce((a, b) => a + b, 0) / scores.value.length)
      : 0

  const summary = await requestSummary()
  
  router.push({
    path: '/result',
    query: {
      avgScore,
      position: position.value,
      feedback_title: summary.feedback_title,
      feedback_message: summary.feedback_message
    }
  })
}

// AI 질문 생성
const questionText = ref('질문을 받아오는 중입니다...')
const loading = ref(false)

async function refreshQuestion() {
  if (questionCount.value >= maxQuestions) return

  loading.value = true
  questionText.value = '질문 생성 중…'

  try {
    const res = await axios.get('http://localhost:8000/api/interview-question', {
      params: { position: position.value },
      withCredentials: true
    })
    questionText.value = res.data.question || '질문을 불러오지 못했습니다.'
  } catch (err) {
    console.error(err)
    questionText.value = '질문을 불러오는 중 오류가 발생했습니다.'
  } finally {
    loading.value = false
  }
}

// 초기 로딩 시 질문 한 번 가져오기
refreshQuestion()
</script>

<style src="./Interview.css" scoped></style>