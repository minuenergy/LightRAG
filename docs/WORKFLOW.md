# LightRAG 워크플로우 (Workflow)

LightRAG는 그래프 기반 RAG(Retrieval-Augmented Generation) 시스템으로, 문서 인덱싱과 쿼리 두 가지 주요 워크플로우를 가지고 있습니다.

---

## 📊 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         LightRAG System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  1. Indexing     │              │  2. Querying     │         │
│  │     Pipeline     │              │     Pipeline     │         │
│  └──────────────────┘              └──────────────────┘         │
│           │                                  │                   │
│           ▼                                  ▼                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Storage Layer                            │       │
│  ├──────────────────────────────────────────────────────┤       │
│  │ • Vector DB (Embeddings)                             │       │
│  │ • Graph DB (Knowledge Graph)                         │       │
│  │ • Key-Value Store (Chunks, Entities, Relations)      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 1. 인덱싱 워크플로우 (Indexing Workflow)

문서를 처리하여 지식 그래프(Knowledge Graph)를 생성하는 과정입니다.

### 플로우차트

```
┌─────────────┐
│   문서 입력   │
│  (Document)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  1. 청킹 (Chunking)  │
│  - Token 기반 분할   │
│  - 의미 단위 유지    │
└──────┬──────────────┘
       │
       ▼
┌────────────────────────────┐
│  2. 엔티티 추출              │
│  (Entity & Relation         │
│   Extraction)               │
│  - LLM을 사용한 추출         │
│  - Entity 식별              │
│  - Relation 식별            │
└──────┬─────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  3. 임베딩 생성              │
│  (Embedding Generation)     │
│  - Entity 임베딩            │
│  - Relation 임베딩          │
│  - Chunk 임베딩             │
└──────┬─────────────────────┘
       │
       ├───────────────────────────┐
       │                           │
       ▼                           ▼
┌─────────────────┐         ┌──────────────────┐
│  4a. 그래프 저장  │         │  4b. 벡터 저장    │
│  (Graph Storage) │         │  (Vector Storage)│
│  - Nodes         │         │  - Embeddings    │
│  - Edges         │         │  - 유사도 검색용  │
└─────────────────┘         └──────────────────┘
       │                           │
       └───────────┬───────────────┘
                   │
                   ▼
           ┌──────────────┐
           │  KV Storage  │
           │  - Chunks    │
           │  - Metadata  │
           └──────────────┘
```

### 세부 단계

#### 1️⃣ **문서 청킹 (Document Chunking)**
- **함수**: `chunking_by_token_size()`
- **처리**:
  - 문서를 토큰 기반으로 분할
  - 청크 크기 제어 (기본값: 설정 가능)
  - 오버랩을 통한 컨텍스트 유지

#### 2️⃣ **엔티티 및 관계 추출 (Entity & Relation Extraction)**
- **함수**: `extract_entities()`
- **처리**:
  - LLM을 사용하여 각 청크에서 엔티티 추출
  - 엔티티 간 관계(Relation) 식별
  - 엔티티 타입 분류 (사람, 장소, 조직 등)
  - Graph 형태로 구조화

#### 3️⃣ **노드 및 엣지 병합 (Node & Edge Merging)**
- **함수**: `merge_nodes_and_edges()`
- **처리**:
  - 중복 엔티티 통합
  - 관계 정규화
  - 지식 그래프 구축

#### 4️⃣ **임베딩 및 저장 (Embedding & Storage)**
- **Vector Storage**: 엔티티, 관계, 청크의 임베딩 저장
- **Graph Storage**: 지식 그래프 구조 저장
- **KV Storage**: 원본 청크 및 메타데이터 저장

---

## 🔍 2. 쿼리 워크플로우 (Query Workflow)

사용자 질문에 대해 지식 그래프를 활용하여 답변을 생성하는 과정입니다.

### 플로우차트

```
┌─────────────┐
│  사용자 쿼리  │
│   (Query)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  1. 쿼리 모드 결정        │
│  - naive: 단순 검색      │
│  - local: 엔티티 중심    │
│  - global: 전체 그래프   │
│  - hybrid: 혼합          │
│  - mix: 전체 혼합        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  2. 듀얼 레벨 검색                    │
│  (Dual-Level Retrieval)              │
├─────────────────────────────────────┤
│                                      │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ High-Level   │  │ Low-Level    │ │
│  │ (Graph-based)│  │ (Chunk-based)│ │
│  └──────┬───────┘  └──────┬───────┘ │
│         │                  │         │
│         ▼                  ▼         │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ 엔티티/관계   │  │ 유사 청크     │ │
│  │ 검색         │  │ 검색         │ │
│  └──────┬───────┘  └──────┬───────┘ │
│         │                  │         │
│         └────────┬─────────┘         │
│                  │                   │
└──────────────────┼───────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  3. 컨텍스트 통합                     │
│  (Context Integration)               │
│  - 그래프 컨텍스트 + 청크 컨텍스트    │
│  - Reranker 적용 (선택)              │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  4. LLM 생성 (Generation)            │
│  - 검색된 컨텍스트 기반 답변 생성     │
│  - 출처 추적 (Citation)              │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  최종 답변   │
│  (Response)  │
└─────────────┘
```

### 쿼리 모드 설명

LightRAG는 5가지 쿼리 모드를 지원합니다:

#### 📌 **1. Naive Mode** (단순 검색)
```
Query → Vector Search (Chunks) → LLM → Response
```
- 가장 기본적인 RAG 방식
- 벡터 유사도 기반 청크 검색
- 지식 그래프 미사용

#### 📌 **2. Local Mode** (로컬 검색)
```
Query → Entity Search → Related Entities & Relations → LLM → Response
```
- 특정 엔티티 중심 검색
- 1-hop 이웃 노드 탐색
- 세부적인 질문에 적합

#### 📌 **3. Global Mode** (글로벌 검색)
```
Query → Graph-level Search → Community Detection → LLM → Response
```
- 전체 그래프 구조 활용
- 고수준 요약 정보 사용
- 광범위한 질문에 적합

#### 📌 **4. Hybrid Mode** (하이브리드)
```
Query → Local + Global → Combined Context → LLM → Response
```
- Local과 Global 결합
- 균형잡힌 검색

#### 📌 **5. Mix Mode** (믹스) ⭐ **권장**
```
Query → Naive + Local + Global → Reranker → LLM → Response
```
- 모든 검색 방식 통합
- Reranker로 최적 컨텍스트 선택
- 최고 성능 (Reranker 설정 시 기본값)

---

## 🏗️ 스토리지 레이어 (Storage Layer)

LightRAG는 3가지 타입의 스토리지를 사용합니다:

```
┌─────────────────────────────────────────────────────┐
│              Storage Architecture                    │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  1. Vector Storage (임베딩 저장)              │   │
│  ├──────────────────────────────────────────────┤   │
│  │  • NanoVectorDB (기본)                        │   │
│  │  • Milvus                                     │   │
│  │  • Qdrant                                     │   │
│  │  • PostgreSQL (pgvector)                     │   │
│  │  • FAISS                                      │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  2. Graph Storage (지식 그래프 저장)          │   │
│  ├──────────────────────────────────────────────┤   │
│  │  • NetworkX (기본)                            │   │
│  │  • Neo4j                                      │   │
│  │  • MongoDB                                    │   │
│  │  • PostgreSQL                                 │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  3. Key-Value Storage (청크/메타데이터 저장)  │   │
│  ├──────────────────────────────────────────────┤   │
│  │  • JSON (기본)                                │   │
│  │  • MongoDB                                    │   │
│  │  • PostgreSQL                                 │   │
│  └──────────────────────────────────────────────┘   │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 🔑 핵심 컴포넌트

### 1. **LLM (Large Language Model)**
- **역할**: 엔티티 추출, 답변 생성
- **요구사항**:
  - 최소 32B 파라미터 권장
  - 32KB+ 컨텍스트 길이
  - 64KB 권장

### 2. **Embedding Model**
- **역할**: 텍스트의 벡터 표현 생성
- **추천 모델**:
  - `BAAI/bge-m3`
  - `text-embedding-3-large`
- **주의**: 인덱싱과 쿼리 시 동일한 모델 사용 필수

### 3. **Reranker (선택)**
- **역할**: 검색 결과 재순위화
- **추천 모델**:
  - `BAAI/bge-reranker-v2-m3`
  - Jina Reranker
- **효과**: Mix 모드에서 성능 대폭 향상

---

## 📈 처리 흐름 요약

### Insert (삽입) 프로세스
```
Document → Chunking → Entity Extraction → Graph Building →
Embedding Generation → Storage (Vector + Graph + KV)
```

### Query (쿼리) 프로세스
```
Query → Mode Selection → Dual-Level Retrieval →
Context Integration → Reranking (optional) → LLM Generation → Response
```

---

## 🎯 최적화 포인트

### 인덱싱 단계
1. **청킹 크기**: 적절한 청크 크기 설정 (너무 작으면 컨텍스트 손실, 너무 크면 정밀도 저하)
2. **병렬 처리**: `max_async`, `max_parallel_insert` 설정으로 속도 향상
3. **LLM 선택**: 인덱싱 단계에서는 reasoning model 비추천

### 쿼리 단계
1. **모드 선택**: 질문 유형에 따라 적절한 모드 선택
2. **Reranker 활용**: Mix 모드 + Reranker 조합 권장
3. **Top-K 조정**: `top_k` 파라미터로 검색 범위 조절

---

## 🔗 참고 자료

- [LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
- [LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
- [LearnOpenCV LightRAG Guide](https://learnopencv.com/lightrag/)

---

## 💡 예제 코드

### 기본 인덱싱
```python
from lightrag import LightRAG

# LightRAG 인스턴스 생성
rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_name="gpt-4o",
    embedding_model_name="text-embedding-3-large"
)

# 문서 삽입
with open("document.txt", "r") as f:
    text = f.read()
    rag.insert(text)
```

### 다양한 모드로 쿼리
```python
# Naive 모드
response = rag.query("질문", mode="naive")

# Local 모드
response = rag.query("질문", mode="local")

# Global 모드
response = rag.query("질문", mode="global")

# Hybrid 모드
response = rag.query("질문", mode="hybrid")

# Mix 모드 (권장)
response = rag.query("질문", mode="mix")
```

---

이 워크플로우를 이해하면 LightRAG의 동작 원리와 최적화 방법을 파악할 수 있습니다.
