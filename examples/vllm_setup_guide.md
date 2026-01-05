# vLLM을 활용한 LightRAG 설정 가이드

이 가이드는 vLLM을 사용하여 LightRAG를 설정하고 테스트하는 방법을 설명합니다.

## 📋 목차

1. [vLLM 서버 실행](#1-vllm-서버-실행)
2. [LightRAG 설정 방법](#2-lightrag-설정-방법)
   - [방법 1: Python 스크립트 사용](#방법-1-python-스크립트-사용)
   - [방법 2: 환경 변수 (.env) 사용](#방법-2-환경-변수-env-사용)
   - [방법 3: LightRAG Server (API + WebUI) 사용](#방법-3-lightrag-server-api--webui-사용)
3. [테스트 실행](#3-테스트-실행)
4. [트러블슈팅](#4-트러블슈팅)

---

## 1. vLLM 서버 실행

### 1.1 LLM 서버 시작 (포트 8000)

```bash
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 8192 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --port 8000 \
  --host 0.0.0.0
```

**파라미터 설명:**
- `--tensor-parallel-size 4`: 4개 GPU에 모델 분산
- `--gpu-memory-utilization 0.30`: GPU 메모리의 30% 사용
- `--max-model-len 8192`: 최대 컨텍스트 길이
- `--max-num-batched-tokens 2048`: 배치 처리 토큰 수
- `--max-num-seqs 8`: 동시 처리 시퀀스 수
- `--port 8000`: 서비스 포트
- `--host 0.0.0.0`: 모든 인터페이스에서 접근 허용

### 1.2 Embedding 서버 시작 (포트 8001)

```bash
vllm serve BAAI/bge-m3 \
  --port 8001 \
  --gpu-memory-utilization 0.1 \
  --host 0.0.0.0
```

**파라미터 설명:**
- `BAAI/bge-m3`: 임베딩 모델 (1024차원)
- `--port 8001`: 서비스 포트
- `--gpu-memory-utilization 0.1`: GPU 메모리의 10% 사용

### 1.3 서버 확인

서버가 정상적으로 실행되었는지 확인:

```bash
# LLM 서버 확인
curl http://localhost:8000/v1/models

# Embedding 서버 확인
curl http://localhost:8001/v1/models
```

---

## 2. LightRAG 설정 방법

### 방법 1: Python 스크립트 사용

제공된 `lightrag_vllm_demo.py` 파일을 사용합니다:

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# vLLM 서버 설정
VLLM_LLM_HOST = "http://localhost:8000/v1"
VLLM_EMBEDDING_HOST = "http://localhost:8001/v1"

async def llm_model_func(prompt, system_prompt=None, **kwargs):
    return await openai_complete_if_cache(
        "openai/gpt-oss-20b",
        prompt,
        system_prompt=system_prompt,
        api_key="EMPTY",  # vLLM은 API key 불필요
        base_url=VLLM_LLM_HOST,
        **kwargs,
    )

async def embedding_func(texts: list[str]):
    return await openai_embed(
        texts,
        model="BAAI/bge-m3",
        api_key="EMPTY",
        base_url=VLLM_EMBEDDING_HOST,
    )

# LightRAG 초기화
rag = LightRAG(
    working_dir="./vllm_test",
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,  # BGE-M3의 임베딩 차원
        max_token_size=8192,
        func=embedding_func,
    ),
)
```

### 방법 2: 환경 변수 (.env) 사용

`.env` 파일을 생성하고 다음 설정을 추가:

```bash
###########################
### LLM Configuration
###########################
LLM_BINDING=openai
LLM_MODEL=openai/gpt-oss-20b
LLM_BINDING_HOST=http://localhost:8000/v1
LLM_BINDING_API_KEY=EMPTY

# OpenAI 호환 API 설정
OPENAI_LLM_TEMPERATURE=0.7
OPENAI_LLM_MAX_TOKENS=4096

###########################
### Embedding Configuration
###########################
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://localhost:8001/v1
EMBEDDING_BINDING_API_KEY=EMPTY
EMBEDDING_TOKEN_LIMIT=8192
EMBEDDING_SEND_DIM=false

###########################
### Storage Configuration
###########################
WORKING_DIR=./vllm_rag_storage

# 기본 스토리지 (간단한 테스트용)
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage

###########################
### Query Configuration
###########################
ENABLE_LLM_CACHE=true
TOP_K=40
CHUNK_TOP_K=20
MAX_ENTITY_TOKENS=6000
MAX_RELATION_TOKENS=8000
MAX_TOTAL_TOKENS=30000

###########################
### Concurrency Configuration
###########################
MAX_ASYNC=4
MAX_PARALLEL_INSERT=2
EMBEDDING_FUNC_MAX_ASYNC=8
EMBEDDING_BATCH_NUM=10

###########################
### Document Processing
###########################
ENABLE_LLM_CACHE_FOR_EXTRACT=true
SUMMARY_LANGUAGE=Korean
CHUNK_SIZE=1200
CHUNK_OVERLAP_SIZE=100
```

그 다음 Python 스크립트에서 환경 변수를 로드:

```python
from dotenv import load_dotenv
load_dotenv()

from lightrag import LightRAG

# .env 파일의 설정이 자동으로 적용됨
rag = LightRAG(
    working_dir=os.getenv("WORKING_DIR", "./vllm_rag_storage")
)
```

### 방법 3: LightRAG Server (API + WebUI) 사용

LightRAG Server를 사용하면 Web UI와 REST API로 사용할 수 있습니다:

#### 3.1 .env 파일 설정

위의 "방법 2"의 `.env` 파일을 사용하고 다음 설정 추가:

```bash
###########################
### Server Configuration
###########################
HOST=0.0.0.0
PORT=9621
WEBUI_TITLE='LightRAG with vLLM'
WEBUI_DESCRIPTION="Graph-based RAG with vLLM Backend"
```

#### 3.2 서버 실행

```bash
# LightRAG Server 시작
lightrag-server

# 또는 Docker를 사용하는 경우
docker compose up
```

#### 3.3 접속

- **Web UI**: http://localhost:9621
- **API Documentation**: http://localhost:9621/docs

---

## 3. 테스트 실행

### 3.1 데모 스크립트 실행

```bash
cd /home/user/LightRAG
python examples/lightrag_vllm_demo.py
```

### 3.2 예상 출력

```
============================================================
LightRAG + vLLM Demo
============================================================
LLM Server: http://localhost:8000/v1
LLM Model: openai/gpt-oss-20b
Embedding Server: http://localhost:8001/v1
Embedding Model: BAAI/bge-m3
Embedding Dimension: 1024
============================================================

[1/4] Initializing RAG instance...
✓ RAG instance initialized

[2/4] Testing embedding function...
✓ Test text: This is a test string for embedding.
✓ Detected embedding dimension: 1024

[3/4] Inserting sample document...
✓ Document inserted successfully

[4/4] Testing queries...
...
```

### 3.3 대화형 쿼리 테스트

```python
import asyncio
from lightrag import LightRAG, QueryParam

async def test_query():
    rag = LightRAG(working_dir="./vllm_test")
    await rag.initialize_storages()

    # 다양한 모드로 쿼리
    modes = ["naive", "local", "global", "hybrid", "mix"]

    for mode in modes:
        print(f"\n=== {mode.upper()} Mode ===")
        result = await rag.aquery(
            "LightRAG의 주요 기능은 무엇인가요?",
            param=QueryParam(mode=mode)
        )
        print(result)

    await rag.finalize_storages()

asyncio.run(test_query())
```

---

## 4. 트러블슈팅

### 4.1 Connection Error

**문제**: `Connection refused` 에러 발생

**해결책**:
```bash
# vLLM 서버가 실행 중인지 확인
ps aux | grep vllm

# 포트가 사용 중인지 확인
lsof -i :8000
lsof -i :8001

# vLLM 로그 확인
# (vLLM 실행 터미널에서 에러 메시지 확인)
```

### 4.2 Out of Memory Error

**문제**: GPU 메모리 부족

**해결책**:
```bash
# GPU 메모리 사용률 감소
vllm serve openai/gpt-oss-20b \
  --gpu-memory-utilization 0.20 \  # 30% → 20%로 감소
  --max-model-len 4096 \            # 8192 → 4096으로 감소
  ...

# 또는 더 작은 모델 사용
vllm serve openai/gpt-oss-7b ...
```

### 4.3 Embedding Dimension Mismatch

**문제**: `Embedding dimension mismatch` 에러

**해결책**:
```python
# BGE-M3의 정확한 임베딩 차원 확인
import requests

response = requests.post(
    "http://localhost:8001/v1/embeddings",
    json={"input": "test", "model": "BAAI/bge-m3"}
)
dim = len(response.json()["data"][0]["embedding"])
print(f"Actual embedding dimension: {dim}")

# LightRAG 설정에 정확한 차원 사용
embedding_func = EmbeddingFunc(
    embedding_dim=dim,  # 확인된 차원 사용
    ...
)
```

### 4.4 Slow Response

**문제**: 쿼리 응답이 느림

**해결책**:
```bash
# .env 파일에서 동시성 설정 조정
MAX_ASYNC=8              # 4 → 8로 증가
MAX_PARALLEL_INSERT=4    # 2 → 4로 증가
EMBEDDING_FUNC_MAX_ASYNC=16  # 8 → 16으로 증가

# vLLM 서버의 배치 크기 증가
vllm serve ... \
  --max-num-seqs 16 \  # 8 → 16
  ...
```

### 4.5 Model Not Found

**문제**: vLLM이 모델을 찾지 못함

**해결책**:
```bash
# 모델을 미리 다운로드
huggingface-cli download openai/gpt-oss-20b
huggingface-cli download BAAI/bge-m3

# 또는 로컬 경로 지정
vllm serve /path/to/local/model ...
```

---

## 📚 추가 리소스

- [vLLM Documentation](https://docs.vllm.ai/)
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

---

## ✅ 체크리스트

설정이 완료되었는지 확인하세요:

- [ ] vLLM LLM 서버 실행 (포트 8000)
- [ ] vLLM Embedding 서버 실행 (포트 8001)
- [ ] 서버 응답 확인 (`curl` 테스트)
- [ ] `.env` 파일 설정 또는 Python 스크립트 작성
- [ ] 임베딩 차원 확인 (BGE-M3: 1024)
- [ ] 데모 스크립트 실행 성공
- [ ] 쿼리 테스트 성공

모든 항목이 체크되었다면 준비 완료입니다! 🎉
