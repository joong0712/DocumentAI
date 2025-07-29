import os
import re
import asyncio
import nest_asyncio
import fitz
from time import time
import traceback
import threading
import hashlib
from datetime import datetime

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    session,
    jsonify,
    g,
)
from flask_socketio import SocketIO, emit, join_room, leave_room
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)  # 비밀번호 해싱
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)  # Flask-Login

from pymongo import MongoClient  # pymongo 임포트
from bson.objectid import ObjectId  # MongoDB ObjectId 사용

# LangChain 관련 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from markupsafe import Markup


nest_asyncio.apply()  # asyncio 중첩 실행 허용

load_dotenv()  # .env 파일에서 환경 변수 로드

app = Flask(__name__)
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "your_super_secret_key_change_me_in_production_really_it_is_important_for_security",
)  # 강력한 SECRET_KEY 사용

# ───[ PDF 업로드 폴더: 기존 ]───────────────────────────────────
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("output", exist_ok=True)  # output 폴더는 사용되지 않는다면 제거 가능

# ───[ 프로필 이미지 업로드 폴더: 신규 ]────────────────────────
PROFILE_UPLOAD_FOLDER = os.path.join(app.static_folder, "profile_images")
os.makedirs(PROFILE_UPLOAD_FOLDER, exist_ok=True)
app.config["PROFILE_UPLOAD_FOLDER"] = PROFILE_UPLOAD_FOLDER

# ───[ 허용 이미지 확장자 검사 ]──────────────────────────────────
ALLWOED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "_" in filename and filename.rsplit("_", 1)[1].lower() in ALLWOED_EXTENSIONS


# ───[ 요약 속도 최적화 ]──────────────────────────────────


def get_chunks(filepath):
    """(최적화) PyMuPDF로 전체 텍스트를 읽고 Document 객체 리스트로 분할"""
    print(f"PyMuPDF로 '{filepath}' 파일 처리 시작...")
    try:
        doc = fitz.open(filepath)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        print(f"PDF 읽기 오류 {filepath}: {e}")
        return []

    if not full_text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(full_text)
    return [Document(page_content=text) for text in text_chunks]


_vectorstore_cache: dict[str, FAISS] = {}


# ---------- pdf 해시 함수 ---------------
def get_file_hash(filepath):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


# 전역 FAISS 인덱스 캐시
def get_vectorstore(filepath, chunks):
    """(최적화) FAISS 인덱스를 파일로 저장/로드하여 캐싱"""
    cache_folder = "./faiss_cache"
    safe_basename = "".join(
        c for c in os.path.basename(filepath) if c.isalnum() or c in ("_", "-")
    ).rstrip()
    faiss_index_path = os.path.join(cache_folder, f"{safe_basename}.faiss")

    if faiss_index_path in _vectorstore_cache:
        print("✅ 인메모리 캐시에서 FAISS 인덱스 로드.")
        return _vectorstore_cache[faiss_index_path]

    if os.path.exists(faiss_index_path):
        try:
            print(f"✅ 파일 캐시에서 FAISS 인덱스 로드: {faiss_index_path}")
            embed_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            db = FAISS.load_local(
                faiss_index_path, embed_model, allow_dangerous_deserialization=True
            )
            _vectorstore_cache[faiss_index_path] = db
            return db
        except Exception as e:
            print(f"⚠️ FAISS 인덱스 로드 실패: {e}. 새로 생성합니다.")

    if not chunks:
        raise ValueError("청크가 비어 벡터스토어를 생성할 수 없습니다.")

    print("✨ 캐시 없음. 새로운 FAISS 인덱스 생성 및 저장...")
    embed_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(chunks, embed_model)
    os.makedirs(cache_folder, exist_ok=True)
    db.save_local(faiss_index_path)
    _vectorstore_cache[faiss_index_path] = db
    return db


# --------------- 첫페이지 요약 --------------------
def generate_preview_summary(filepath):
    """(신규) PDF의 첫 페이지만으로 빠른 미리보기 요약을 생성"""
    try:
        doc = fitz.open(filepath)
        first_page_text = doc[0].get_text(sort=True).strip()  # 정렬 옵션 추가
        doc.close()

        if not first_page_text:
            return ""

        prompt = PromptTemplate.from_template(
            "아래 첫 페이지 내용을 읽고, 한국어 친근하고 정중한 존댓말로 2~3문장 이내로 간결하게 요약해주세요:\n\n"
            "---\n"
            "{text}\n"
            "---"
        )
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        chain = prompt | llm

        preview_summary = chain.invoke(
            {"text": first_page_text[:2000]}
        ).content  # 너무 길 경우를 대비해 텍스트 양 제한
        return preview_summary
    except Exception as e:
        print(f"미리보기 요약 생성 중 오류: {e}")
        return ""


def process_full_document_in_background(
    app_context, filepath, relative_path, user_id, filename
):
    """(신규/백그라운드) 전체 문서를 처리하고 완료 시 클라이언트에 신호를 보냄"""
    with app_context:  # 앱 컨텍스트를 사용하여 DB, socketio 등 Flask 리소스에 접근
        print(f"--- ⏳ 백그라운드 작업 시작: {filepath} ---")
        try:
            t_start = time()
            chunks = get_chunks(filepath)
            if not chunks:
                print("백그라운드: 텍스트 추출 실패")
                # 실패 시에도 사용자에게 알릴 수 있습니다.
                socketio.emit(
                    "update_failed",
                    {"message": "PDF에서 텍스트를 추출하지 못했습니다."},
                    room=user_id,
                )
                return

            db = get_vectorstore(filepath, chunks)
            query_for_core_content = (
                f"'{filename}' 문서의 핵심 주제, 주장, 결론은 무엇인가요?"
            )
            core_chunks = db.similarity_search(query_for_core_content, k=5)

            class FinalSummaryAndQuestions(BaseModel):
                summary: str = Field(
                    description="제공된 텍스트 전체의 핵심 내용을 '세 문장 내외'로 요약합니다."
                )
                questions: list[str] = Field(
                    description="텍스트 전체 내용 기반의 흥미로운 질문 목록 (3개)"
                )

            parser = JsonOutputParser(pydantic_object=FinalSummaryAndQuestions)
            prompt = PromptTemplate(
                template="당신은 주어진 문서 내용을 분석하여, 핵심 요약과 질문 3개를 생성하는 AI 어시스턴트입니다. 반드시 아래 JSON 형식에 맞춰 응답해주세요.\n{format_instructions}\n\n[문서 내용]\n{document_content}",
                input_variables=["document_content"],
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )
            llm = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            chain = prompt | llm | parser

            unique_contents = list(dict.fromkeys([c.page_content for c in core_chunks]))
            combined_content = "\n\n---\n\n".join(unique_contents)

            result = chain.invoke({"document_content": combined_content})
            final_summary = result["summary"]
            questions_list = result["questions"]

            questions_list_for_template = [
                {"text": q, "page": 0} for q in questions_list
            ]

            # 1. DB 업데이트
            document_meta_collection.update_one(
                filter={"filepath": relative_path, "user_id": user_id},
                update={
                    "$set": {
                        "summary": final_summary,
                        "questions": questions_list_for_template,
                        "status": "completed",
                    }
                },
                upsert=True,
            )
            print(
                f"--- ✅ 백그라운드 DB 업데이트 완료: {filepath} ({time() - t_start:.2f}s) ---"
            )

            # 🌟 2. (핵심) 작업 완료 신호를 해당 사용자에게만 전송! 🌟
            socketio.emit(
                "summary_updated",
                {
                    "summary": final_summary,
                    "questions": questions_list_for_template,
                    "message": "문서 전체에 대한 분석이 완료되었습니다!",
                },
                room=user_id,
            )  # user_id를 방(room) 이름으로 사용하여 특정 사용자에게만 보냅니다.

            print(f"--- 📡 백그라운드 Socket.IO 신호 전송 완료 (To: {user_id}) ---")

        except Exception as e:
            print(f"--- 🚨 백그라운드 작업 오류: {e} ---")
            document_meta_collection.update_one(
                filter={"filepath": relative_path, "user_id": user_id},
                update={
                    "$set": {"status": "error", "summary": f"처리 중 오류 발생: {e}"}
                },
                upsert=True,
            )
            # 🌟 오류 발생 사실도 사용자에게 알려줍니다.
            socketio.emit(
                "update_failed", {"message": f"오류가 발생했습니다: {e}"}, room=user_id
            )
            traceback.print_exc()


# 🌟 MongoDB 설정 🌟
MONGO_URI = os.getenv(
    "MONGO_URI", "mongodb://localhost:27017/"
)  # .env 파일에서 MONGO_URI 로드 또는 기본값 사용
client = MongoClient(MONGO_URI)
db = client.pdf_chat_db  # 데이터베이스 이름 (예: pdf_chat_db)
chat_history_collection = db.chat_history  # 채팅 기록 컬렉션
document_meta_collection = (
    db.document_meta
)  # 문서 메타데이터 (filepath, summary, questions) 컬렉션
users_collection = db.users  # 🌟 사용자 정보를 저장할 컬렉션 🌟

socketio = SocketIO(app)


@socketio.on("connect")
@login_required
def handle_connect():
    user_id = current_user.get_id()
    join_room(user_id)
    print(
        f"--- 🙋‍♂️ Client connected: {request.sid}, User: {current_user.username}, Room: {user_id} ---"
    )


@socketio.on("disconnect")
def handle_disconnect():
    # 필요하다면 방에서 나가는 로직을 추가할 수 있지만, 보통은 자동 처리됩니다.
    print(f"--- 🤦‍♂️ Client disconnected: {request.sid} ---")


# 🌟 Flask-Login 초기화 🌟
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # 로그인되지 않은 사용자가 @login_required 페이지 접근 시 리다이렉트할 라우트


# 🌟 User 모델 정의 (MongoDB와 연동) 🌟
class User(UserMixin):
    def __init__(self, user_data):
        self._id = user_data["_id"]
        self.username = user_data["username"]
        self.password_hash = user_data["password_hash"]
        self.profile_image = user_data.get("profile_image", "default-profile.png")

    def get_id(self):
        # Flask-Login은 사용자 ID를 문자열로 기대하므로 ObjectId를 문자열로 변환
        return str(self._id)

    def set_password(self, password):
        """비밀번호를 해싱하여 저장합니다."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """저장된 해시된 비밀번호와 입력된 비밀번호를 비교합니다."""
        return check_password_hash(self.password_hash, password)


# 🌟 Flask-Login이 사용자 ID를 기반으로 User 객체를 로드하는 함수 🌟
@login_manager.user_loader
def load_user(user_id):
    # MongoDB에서 ObjectId를 사용하여 사용자 문서 찾기
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None


# 🌟 초기 사용자 데이터베이스 설정 및 테스트 사용자 생성 🌟
# 앱 시작 시 한 번만 실행
def initialize_database():
    with app.app_context():  # app.app_context() 대신 app.app_content()로 수정 (오타 수정)
        if users_collection.find_one({"username": "testuser"}) is None:
            test_user_data = {
                "username": "testuser",
                "password_hash": generate_password_hash(
                    "password123"
                ),  # 테스트 비밀번호 'password123'
            }
            users_collection.insert_one(test_user_data)
            print(
                "테스트 사용자 'testuser' (비밀번호: password123)가 MongoDB에 생성되었습니다."
            )


# --- 비동기 요약 함수 ---
async def summarize_chunk(llm, prompt_template, chunk, idx):
    chain = LLMChain(llm=llm, prompt=prompt_template)
    raw = await chain.arun(chunk.page_content)
    page_meta = chunk.metadata.get("page")

    if page_meta is not None:
        page_num = page_meta + 1
    else:
        page_num = chunk.metadata.get("page_number") or (idx + 1)
    return (
        f"{raw.strip()} "
        f'<a href="#" onclick="goToPage({page_num});return false;">'
        f"(페이지 {page_num})</a>"
    )


async def parallel_summary(llm, prompt_template, chunks):
    tasks = [
        summarize_chunk(llm, prompt_template, chunk, idx)
        for idx, chunk in enumerate(chunks)
    ]
    result = await asyncio.gather(*tasks)
    return "\n".join(result)


# --- 질문 생성 함수 ---
def generate_question(summary_text):
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
        다음 요약 내용을 기반으로 사람들이 자주 물어볼만한 질문 1가지를 한국어로 알려줘.
        각 질문은 간결하게 작성해줘.
        줄바꿈으로 구분해줘.

        요약:
        {summary}
        """,
    )
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(summary_text)
    question = result.strip().strip("-•* 0123456789.")
    return question


# --- 프롬프트 템플릿 ---
simple_prompt = PromptTemplate(
    input_variables=["text"], template="다음 문서를 한국어로 간단히 요약해줘:\n\n{text}"
)

detailed_prompt = PromptTemplate(
    input_variables=["text"],
    template="다음 문서를 한국어로 자세하고 핵심적으로 요약해줘:\n\n{text}",
)


# --- Jinja2 필터: 줄바꿈을 <br> 태그로 변환 ---
@app.template_filter("nl2br")
def nl2br_filter(s):
    if not isinstance(s, str):
        return s
    return Markup(re.sub(r"\r?\n", "<br>\n", s))


# --- Flask 라우트 ---


# 🌟 루트 URL: 로그인 상태에 따라 로그인 페이지 또는 PDF Chat 페이지로 리다이렉트 🌟
@app.route("/")
def root():
    if current_user.is_authenticated:
        return redirect(url_for("pdf_chat_page"))
    return redirect(url_for("login"))


# 🌟 로그인 페이지 🌟
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("pdf_chat_page"))

    error = None
    message = request.args.get("message")  # 회원가입 성공 메시지 등을 받을 수 있음

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user_data = users_collection.find_one(
            {"username": username}
        )  # MongoDB에서 사용자 찾기
        if user_data and check_password_hash(user_data["password_hash"], password):
            session.clear()

            login_user(User(user_data))  # Flask-Login으로 사용자 로그인 처리
            return render_template("login.html", error=None)
        error = "아이디 또는 비밀번호가 올바르지 않습니다."
    return render_template("login.html", error=error, message=message)


# 🌟 회원가입 페이지 🌟
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("pdf_chat_page"))

    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # 1) 프로필 이미지 처리
        file = request.files.get("profile_image")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["PROFILE_UPLOAD_FOLDER"], filename)
            file.save(save_path)
        else:
            filename = "default-profile.png"

        if password != confirm_password:
            error = "비밀번호가 일치하지 않습니다."
        elif users_collection.find_one({"username": username}):  # 아이디 중복 확인
            error = "이미 존재하는 아이디입니다."
        else:
            # 새로운 사용자 MongoDB에 저장
            new_user_data = {
                "username": username,
                "password_hash": generate_password_hash(password),
                "profile_image": filename,
            }
            users_collection.insert_one(new_user_data)
            return redirect(
                url_for(
                    "login",
                    message="회원가입이 성공적으로 완료되었습니다. 로그인해주세요.",
                )
            )
    return render_template("login.html", error=error, show_register_modeal=True)


# 🌟 로그아웃 라우트 🌟
@app.route("/logout")
@login_required  # 로그인된 사용자만 로그아웃 가능
def logout():
    logout_user()  # Flask-Login으로 로그아웃 처리
    session.clear()
    return redirect(url_for("login", message="로그아웃되었습니다."))


# 🌟 PDF Chat 메인 페이지 🌟
@app.route("/pdf_chat")
@login_required  # 🌟 로그인된 사용자만 접근 가능 🌟
def pdf_chat_page():
    print(f"--- /pdf_chat 요청 수신 (사용자: {current_user.username}) ---")
    print(
        f"--- [디버깅] /pdf_chat 페이지 로드. 현재 세션 경로: {session.get('filepath')} ---"
    )
    current_filepath = session.get("filepath")  # 현재 세션에 저장된 PDF 파일 경로

    print("current_filepath (세션):", current_filepath)
    print("=== 해당 PDF의 DB 채팅 엔트리 ===")
    for entry in chat_history_collection.find(
        {"pdf_path": current_filepath, "user_id": current_user.get_id()}
    ):
        print(entry)
    for entry in chat_history_collection.find(
        {"pdf_path": current_filepath, "user_id": current_user.get_id()}
    ):
        print("DB entry:", entry)

    # 🌟 MongoDB에서 현재 사용자의 문서 이력 로드 (사용자별로 관리하려면 user_id 필터 추가 필요) 🌟
    # 현재는 세션에 저장된 history를 사용하지만, MongoDB에 저장된 문서 메타데이터에서 가져올 수 있음
    # 예: history_from_db = list(document_meta_collection.find({'user_id': current_user.get_id()}).sort('timestamp', -1).limit(10))
    history_from_db = list(
        document_meta_collection.find({"user_id": current_user.get_id()})
        .sort("timestamp", -1)
        .limit(10)
    )  # 모든 문서 중 최신 10개
    history_filenames = [
        doc.get("display_name", "") for doc in history_from_db
    ]  # 파일 이름만 추출

    # 🌟 MongoDB에서 해당 파일의 채팅 기록 로드 🌟
    current_chat_history = []
    if current_filepath:
        # MongoDB에 저장된 chat_history는 {'role': 'user/ai', 'message': '...', 'pdf_path': '...'} 형태
        db_history = chat_history_collection.find(
            {"pdf_path": current_filepath, "user_id": current_user.get_id()}
        ).sort("timestamp", 1)
        for entry in db_history:
            # 추천질문만 list
            if isinstance(entry["message"], list):
                msg_content = entry["message"]
            elif isinstance(entry["message"], dict):
                msg_content = entry["message"]  # dict는 그대로!
            else:
                msg_content = str(entry["message"])
            current_chat_history.append((entry["role"], msg_content))

        # 🌟 MongoDB에서 문서 메타데이터 로드 (요약, 추천 질문) 🌟
        doc_meta = document_meta_collection.find_one({"filepath": current_filepath})
        if doc_meta:
            # 초기 시스템 메시지는 DB에서 로드된 채팅 기록에 포함되지 않으므로, 여기서 다시 추가
            # 단, 이미 채팅 기록에 시스템 메시지가 있다면 중복 추가 방지
            # (이 로직은 클라이언트 JS에서 처리하는 것이 더 적합할 수 있음)
            if not current_chat_history or current_chat_history[0][0] != "시스템":
                intro_message = f"""
                    안녕하세요! 👋
                    이 문서는 다음과 같은 내용을 담고 있어요:

                    📄 {doc_meta.get('summary', '').strip()}

                    궁금한 내용을 자유롭게 질문해 주세요!
                    """
                current_chat_history.insert(0, ("시스템", intro_message))
                current_chat_history.insert(
                    1, ("추천질문", doc_meta.get("questions", []))
                )

    print(f"PDF Chat 템플릿으로 전달될 채팅 기록 (DB 로드): {current_chat_history}")
    summary = session.get("short_summary", "")
    if summary == "[]":
        summary = ""

    return render_template(
        "index.html",
        filepath=current_filepath,
        summary=summary,  # 현재 세션 요약 (DB에서 로드된 것과 다를 수 있음)
        history=history_filenames,  # MongoDB에서 로드된 파일 이름 목록
        chat_history=current_chat_history,
        recommended_question=session.get(
            "recommended_question", []
        ),  # 현재 세션 추천 질문
        username=current_user.username,  # 🌟 로그인된 사용자 이름 템플릿으로 전달 🌟
    )


# L376 근처의 upload_ajax 함수 전체를 아래 내용으로 교체하세요.


@app.route("/upload_ajax", methods=["POST"])
@login_required
def upload_ajax():
    t0 = time()
    print(f"--- /upload_ajax 요청 수신 (사용자: {current_user.username}) ---")

    if "pdf_file" not in request.files:
        return jsonify({"status": "error", "message": "파일이 없습니다."}), 400

    file = request.files["pdf_file"]
    if file.filename == "":
        return (
            jsonify({"status": "error", "message": "파일을 선택하지 않았습니다."}),
            400,
        )

    filename = secure_filename(file.filename)
    display_name = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    relative_path = os.path.relpath(filepath, app.static_folder)
    preview_summary = generate_preview_summary(filepath)
    print(f"[프로파일] 파일 저장: {time() - t0:.3f}s")

    # -----pdf 해시값 계산 -------
    file_hash = get_file_hash(filepath)

    doc_in_db = document_meta_collection.find_one(
        {"user_id": str(current_user.get_id()), "file_hash": file_hash}
    )

    if doc_in_db:
        status = "same"
        upload_msg = "이미 해당 문서로 대화한 이력이 있습니다. <br> 기존 대화를 불러오겠습니다."
        response = {
            "message": upload_msg,
            "status": status,
            "filepath": doc_in_db.get("filepath", ""),
            "filename": filename,
        }
    else:
        status = "new"
        upload_msg = (
            "새 문서가 추가되었습니다. 새로운 대화를 시작합니다.\n\n"
        "첫 페이지만 먼저 요약했어요! 나머지는 백그라운드에서 처리 중입니다 😊\n"
        "나머지 페이지도 백그라운드에서 분석 중이며, 곧 전체 결과를 알려드릴게요!"
        )

        document_meta_collection.update_one(
            filter={"user_id": str(current_user.get_id()), "file_hash": file_hash},
            update={
                "$set": {
                    "filename": filename,
                    "display_name": display_name,
                    "filepath": relative_path,
                    "file_hash": file_hash,
                    "summary": (
                        preview_summary if isinstance(preview_summary, str) else ""
                    ),
                    "questions": [],
                    "timestamp": datetime.now(),
                    "user_id": str(current_user.get_id()),
                    "status": "processing",
                }
            },
            upsert=True,
        )
    response = {
        "summary": preview_summary,
        "message": upload_msg,
        "status": status,
        "filepath": relative_path,
        "filename": filename,
    }
    print(f"[UPLOAD] 업로드된 파일명: {filename}")
    print(f"[UPLOAD] 상대 경로(relative_path): {relative_path}")

    session["filepath"] = relative_path
    print(f"[UPLOAD] 세션에 저장한 filepath: {session.get('filepath')}")
    print(
        f"--- [디버깅] /upload_ajax: 세션에 저장된 파일 경로: {session.get('filepath')} ---"
    )

    # ── "첫 페이지만" 요약하여 즉시 응답 (🚀) ────────────────
    t_preview = time()
    preview_summary = generate_preview_summary(filepath)

    if isinstance(preview_summary, list) or preview_summary == "[]":
        preview_summary = ""
    print(f"[프로파일] 미리보기 요약 생성: {time() - t_preview:.3f}s")

    # ── 백그라운드에서 전체 문서 처리 시작 ──────────────────
    # ⚠️ 경고: threading은 간단한 시연용입니다. 실제 프로덕션 환경에서는
    # 반드시 Celery나 Dramatiq 같은 전문 작업 큐를 사용해야 합니다.
    background_thread = threading.Thread(
        target=process_full_document_in_background,
        args=(
            app.app_context(),
            filepath,
            relative_path,
            current_user.get_id(),
            filename,
        ),
    )
    background_thread.start()
    print(f"🚀 [프로파일] 사용자에게 즉시 응답: {time() - t0:.3f}s")
    return jsonify(response)


# 🌟 Socket.IO 이벤트 핸들러 🌟
@socketio.on("send_question")
@login_required  # 🌟 Socket.IO 이벤트도 로그인된 사용자만 가능 🌟
def handle_send_question(data):
    user_question = data.get("user_question")
    pdf_path_from_client = data.get("pdf_path")
    session_id = request.sid
    user_id = current_user.get_id()

    print(
        f"--- SocketIO 질문 수신 (SID: {session_id}, 사용자: {current_user.username}) ---"
    )
    print(f"클라이언트로부터 받은 질문: '{user_question}'")
    print(f"클라이언트로부터 받은 PDF 경로: '{pdf_path_from_client}'")

    # 1) PDF 경로 유효성 검사
    if not pdf_path_from_client:
        error_msg = (
            "오류: PDF 파일 경로가 유효하지 않습니다. PDF를 먼저 업로드해 주세요."
        )
        print(f"오류: {error_msg}")
        emit("ai_response_chunk", {"chunk": error_msg}, room=session_id)
        emit("ai_response_end", {"text": error_msg, "page": None}, room=session_id)
        return

    full_pdf_path = os.path.join(app.static_folder, pdf_path_from_client)
    print(f"서버에서 사용할 PDF 절대 경로: '{full_pdf_path}'")

    # 2) 실제 파일 존재 여부 검사
    if not os.path.exists(full_pdf_path):
        error_msg = f"오류: 서버에 PDF 파일이 없습니다: '{full_pdf_path}'. 다시 업로드해 주세요."
        print(f"오류: {error_msg}")
        emit("ai_response_chunk", {"chunk": error_msg}, room=session_id)
        emit("ai_response_end", {"text": error_msg, "page": None}, room=session_id)
        return

    # 3) 사용자 질문 MongoDB 저장
    chat_history_collection.insert_one(
        {
            "user_id": user_id,
            "pdf_path": pdf_path_from_client,
            "role": "user",
            "message": user_question,
            "timestamp": datetime.now(),
        }
    )
    print(f"사용자 질문 MongoDB에 저장됨: '{user_question}'")

    # 4) 백그라운드 태스크로 답변 생성/스트리밍 시작
    socketio.start_background_task(
        target=process_question_and_stream,
        session_id=session_id,
        user_question=user_question,
        full_pdf_path=full_pdf_path,
        pdf_path_for_db=pdf_path_from_client,
        user_id_for_db=user_id,
    )


def process_question_and_stream(
    session_id, user_question, full_pdf_path, pdf_path_for_db, user_id_for_db
):
    print(
        f"--- 백그라운드 태스크 시작 (SID: {session_id}, 사용자: {user_id_for_db}) ---"
    )
    full_answer = ""
    source_page = None

    try:
        # PDF 로드 & 청크 분할
        loader = PyPDFLoader(full_pdf_path)
        docs = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        ).split_documents(loader.load())
        print(f"문서 로드 및 청크 분할 완료. 청크 수: {len(docs)}")

        # FAISS 벡터 DB 생성 & 검색
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        retriever_docs = retriever.get_relevant_documents(user_question)
        context = "\n".join(doc.page_content for doc in retriever_docs)
        print(f"관련 문서 검색 완료. 컨텍스트 길이: {len(context)}")

        # 소스 페이지 추출
        source_page = None
        if retriever_docs and getattr(retriever_docs[0], "metadata", None):
            source_page = retriever_docs[0].metadata.get("page")
        print(f"추출된 페이지 번호: {source_page}")

        # LLM 스트리밍 답변
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            다음 문서를 기반으로 한국어로 답변해줘.

            문서 내용:
            {context}

            질문:
            {question}
            """,
        )
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
            streaming=True,
        )
        qa_chain = LLMChain(llm=llm, prompt=prompt_template)

        for chunk in qa_chain.stream({"context": context, "question": user_question}):
            chunk_text = chunk.get("text", "")
            full_answer += chunk_text
            socketio.emit(
                "ai_response_chunk",
                {"chunk": chunk_text, "avatar": "/static/img/ai-avatar.png"},
                room=session_id,
            )
            socketio.sleep(0.02)

        print(f"스트리밍 완료. 전체 답변 길이: {len(full_answer)}")

        ai_answer = full_answer
        page_num = source_page + 1 if source_page is not None else None

        # 성공 답변 MongoDB 저장
        chat_history_collection.insert_one(
            {
                "user_id": user_id_for_db,
                "pdf_path": pdf_path_for_db,
                "role": "ai",
                "message": {
                    "text": full_answer,
                    "avatar": "/static/img/ai-avatar.png",
                    "page": page_num,
                },
                "timestamp": datetime.now(),
            }
        )
        print("AI 답변 MongoDB 저장 완료.")

        # 스트리밍 종료 이벤트 (text + page)
        socketio.emit(
            "ai_response_end",
            {
                "full_text": full_answer,
                "page": source_page + 1 if source_page is not None else None,
            },
            room=session_id,
        )
        print("ai_response_end 이벤트 전송 완료.")

    except Exception as e:
        # 에러 처리
        error_msg = f"오류가 발생하여 답변을 생성할 수 없습니다: {e}"
        print(f"예외 발생: {error_msg}")
        traceback.print_exc()

        # 에러 MongoDB 저장
        chat_history_collection.insert_one(
            {
                "user_id": user_id_for_db,
                "pdf_path": pdf_path_for_db,
                "role": "ai",
                "message": {"text": error_msg, "page": None},
                "timestamp": datetime.now(),
            }
        )
        print("에러 메시지 MongoDB 저장 완료.")

        # 에러 청크 전송
        socketio.emit("ai_response_chunk", {"chunk": error_msg}, room=session_id)

        # 에러 스트리밍 종료 (text + page=None)
        socketio.emit(
            "ai_response_end", {"text": error_msg, "page": None}, room=session_id
        )
        print("에러용 ai_response_end 이벤트 전송 완료.")


# Flask 앱 실행
if __name__ == "__main__":
    initialize_database()
    socketio.run(
        app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True, port=5000
    )
