import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 환경 변수 로드 및 페이지 설정
load_dotenv()
st.set_page_config(page_title="보험 약관 챗봇", page_icon="📄", layout="centered")

# OpenAI API Key 확인 안내 (UI)
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ `.env` 파일에 `OPENAI_API_KEY`가 설정되어 있지 않습니다.")
    st.stop()

# 2. PDF 처리 및 벡터스토어 생성 함수 (캐싱 적용)
@st.cache_resource
def init_vectorstore(pdf_path):
    if not os.path.exists(pdf_path):
        return None

    # PDF 텍스트 추출
    pdf_doc = fitz.open(pdf_path)
    text = ""
    for page in pdf_doc:
        text += page.get_text()
        text += '\n\n'

    # 청킹
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitters.split_text(text)

    # 벡터스토어 생성 및 반환
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 파일 경로 (코드가 있는 폴더에 PDF가 함께 있어야 합니다)
PDF_FILE_PATH = "9회주는 암보험Plus_해약환급금 미지급형.pdf"

# 메인 UI 타이틀
st.title("📄 보험 약관 Q&A 시스템")
st.write("`9회주는 암보험Plus` 약관을 기반으로 AI가 답변해 드립니다.")
st.markdown("---")

# 백그라운드에서 벡터스토어 초기화
with st.spinner("📦 약관 데이터를 분석하고 벡터 데이터베이스를 빌드하는 중입니다..."):
    vectorstore = init_vectorstore(PDF_FILE_PATH)

if vectorstore is None:
    st.error(f"❌ '{PDF_FILE_PATH}' 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    st.stop()
else:
    st.success("✅ 약관 데이터 분석 완료!")

# 3. 사용자 질문 입력 및 처리
query = st.text_input(
    "💡 궁금하신 내용을 입력하세요:", 
    value="암보험의 월 납입해야할 1회 보험료가 얼마인가요?",
    placeholder="예: 암 진단비 지급 조건이 어떻게 되나요?"
)

if st.button("질문하기", type="primary"):
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        with st.spinner("🔍 약관에서 관련 내용을 찾아 답변을 생성하고 있습니다..."):
            try:
                # 유사도 검색 (상위 4개 청크)
                docs_with_score = vectorstore.similarity_search_with_score(query, k=4)

                # 컨텍스트 병합
                context = ""
                for doc, score in docs_with_score:
                    context += doc.page_content
                    context += "\n\n"

                # 체인 및 LLM 설정 (gpt-4o 모델 사용)
                llm = ChatOpenAI(temperature=0, model='gpt-4o')

                template = """다음 배경 지식을 사용해서 질문에 대답해 주세요.
                배경 지식에 없는 내용이거나, 개인별로 달라지는 정보(예: 개인 보험료)라면 약관 내 일반적인 기준을 설명하거나 "제공된 문서에서 개인 맞춤 정보를 찾을 수 없습니다."라고 답변해 주세요.

                배경지식
                {context}
                ============
                질문
                {question}"""

                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm | StrOutputParser()

                # 체인 실행
                inputs = {"context": context, 'question': query}
                response = chain.invoke(inputs)

                # 결과 출력
                st.markdown("### 🤖 AI 답변")
                st.info(response)

                # (선택 사항) 참고한 약관 본문 보여주기 (토글 형식)
                with st.expander("📚 답변에 참고한 약관 본문 보기"):
                    for idx, (doc, score) in enumerate(docs_with_score):
                        st.markdown(f"**[참고 문단 {idx+1}]** (유사도 점수: {score:.4f})")
                        st.code(doc.page_content, language="text")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
