# RAG (Retrieval-Augmented Generation) 개념 및 필수 용어 정리

## 1. RAG란?
**RAG (검색증강생성)**은 대형 언어 모델(LLM)의 답변 생성 과정에 외부 지식을 검색하여 추가함으로써, 더 정확하고 최신성 있는 답변을 만드는 기법입니다.

---

## 2. 한 줄 핵심 요약
- **RAG (Retrieval-Augmented Generation / 검색증강생성)**: LLM이 외부 지식을 검색해 답변의 정확성과 최신성을 높이는 기법.
- **Document Loader (문서 로더)**: 다양한 형식의 데이터를 불러와 문서 객체로 변환하는 도구.
- **Chunking (청크 분할)**: 긴 문서를 작은 텍스트 조각으로 나누어 검색과 임베딩 효율을 높이는 작업.
- **Embedding (임베딩)**: 텍스트를 의미 기반의 고차원 벡터로 변환하는 표현 방법.
- **Vector Store (벡터 저장소)**: 임베딩 벡터를 저장하고 유사도 검색을 수행하는 특수 데이터베이스.
- **Retriever (리트리버 / 검색기)**: 쿼리에 맞는 관련 문서 조각을 벡터 스토어 등에서 찾아오는 모듈.
- **LLM (Large Language Model / 대규모 언어 모델)**: 대규모 학습 데이터로 훈련된 자연어 처리·생성 모델.
- **Prompt Template (프롬프트 템플릿)**: LLM에 줄 프롬프트의 형식을 미리 정의해 변수만 채우는 틀.
- **Chain / Pipeline (체인 / 파이프라인)**: 여러 RAG 구성 요소를 순차적으로 연결해 하나의 작업 흐름으로 만든 구조.
- **Memory (메모리 / 대화 기억)**: 이전 대화나 상태를 저장해 후속 요청 시 문맥을 유지하는 기능.
- **Query (쿼리 / 질의)**: 사용자가 입력한 질문이나 검색어.
- **Contextual Compression (맥락 압축)**: 검색된 문서에서 질문과 관련된 핵심 내용만 추려 컨텍스트를 압축하는 기법.

---

## 3. 간단한 LangChain 기반 RAG 예시
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 문서 로드 및 분할
loader = TextLoader("example.txt", encoding="utf8")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 2. 벡터 스토어 생성
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 3. LLM + QA 체인 구성
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# 4. 질의 실행
query = "예시 지식 베이스에 대한 질문을 여기에 넣으세요."
result = qa_chain.run(query)
print("답변:", result)
```

---
이 파일은 RAG 개념과 용어, 핵심 요약, 그리고 LangChain 예시 코드를 포함하고 있습니다.
