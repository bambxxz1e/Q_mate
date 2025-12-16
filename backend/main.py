# 실행 방법
# & D:\Q_mate\.venv\Scripts\Activate.ps1
# cd backend
# uvicorn main:app --reload --port 8000

from fastapi import FastAPI, UploadFile, File, Query, Form
import google.generativeai as genai
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
from pydantic import BaseModel
from typing import List

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini 설정 ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# --- Whisper 모델 로드 (최초 한 번) ---
whisper_model = whisper.load_model("base")

# -- 질문 생성 --
@app.get("/api/interview-question")
async def interview_question(position: str = Query("일반")):
    prompt = f"너는 전문 면접관이야. 다른 불필요한 대답하지말고 {position} 면접 질문 한 문장 만들어서 그것만 출력해줘."
    response = model.generate_content(prompt)
    question = response.text.strip()

    print(f"Generated question for {position}:", question)
    return {"question": question}

# --- 답변 평가 ---
@app.post("/api/evaluate-answer")
async def evaluate_answer(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    # 1. 업로드된 파일을 임시 파일로 저장
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 2. Whisper로 음성 → 텍스트
    result = whisper_model.transcribe(tmp_path, language="ko")
    answer_text = result["text"].strip()
    print("사용자 답변: ", answer_text)

    # 3. AI 평가
    prompt = f"""
    너는 면접관이야.
    질문: {question}
    지원자 답변: {answer_text}
    1) 답변이 질문을 얼마나 잘 이해하고 정확히 답했는지 0~100 점으로 숫자만 출력해줘. 점수는 최대한 후하게 잘 주었으면 좋겠어.
    2) 답변 개선을 위한 간단한 조언이 있다면 한문장도 출력해줘.
    반드시 아래 형식으로 출력해줘: 점수/조언 (예시: 85/조금 더 또박또박 말해보아요.)
    """
    response = model.generate_content(prompt)
    
    try:
        score_str, advice = response.text.split("/", 1)  # '/' 기준으로 나누기
        score = int(score_str.strip())  # 숫자만
        advice = advice.strip()  # 조언
    except Exception:
        score = 0
        advice = ""

    print("AI가 판단한 점수:", score)
    print("AI 조언:", advice)

    # score와 advice 둘 다 반환
    return {"score": score, "advice": advice}

# -- 피드백 요약 --
class SummaryRequest(BaseModel):
    advices: List[str]

@app.post("/api/interview-summary")
async def interview_summary(req: SummaryRequest):
    joined = "\n".join([f"- {a}" for a in req.advices])

    prompt = f"""
    너는 전문 면접관이야.

    아래는 면접 중 AI가 각 질문마다 생성한 조언들이야:
    {joined}

    이 조언들을 종합해서
    면접 결과 화면에 보여줄 최종 피드백을 작성해줘.

    규칙:
    1) 제목은 10~15자 이내의 간결한 문장
    2) 내용은 한 문장, 부드러운 톤의 개선 조언
    반드시 아래 형식으로 출력해줘: 제목/내용 (출력 예시:바디 랭귀지 관리/손동작을 줄이고 시선과 말의 흐름을 안정시키면 더 신뢰감 있는 인상을 줄 수 있어요.)
    """

    response = model.generate_content(prompt)
    raw = response.text.strip()
    print("AI 종합 평가:", raw)

    if "/" in raw:
        title, message = raw.split("/", 1)
        title = title.strip()
        message = message.strip()
    else:
        # fallback
        title = "AI 종합 피드백"
        message = raw

    return {
        "feedback_title": title,
        "feedback_message": message
    }