"""
LightRAG with vLLM Example
===========================

This example demonstrates how to use LightRAG with vLLM-deployed models:
1. LLM: openai/gpt-oss-20b (port 8000)
2. Embedding: BAAI/bge-m3 (port 8001)

Prerequisites:
--------------
1. Start LLM server:
   vllm serve openai/gpt-oss-20b --tensor-parallel-size 4 --gpu-memory-utilization 0.30 \
        --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 8 --port 8000

2. Start Embedding server:
   vllm serve BAAI/bge-m3 --port 8001 --gpu-memory-utilization 0.1 --host 0.0.0.0

Usage:
------
python examples/lightrag_vllm_demo.py
"""

import os
import asyncio
import inspect
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# vLLM 서버 설정
VLLM_LLM_HOST = "http://localhost:8000/v1"  # LLM 서버 (gpt-oss-20b)
VLLM_EMBEDDING_HOST = "http://localhost:8001/v1"  # Embedding 서버 (bge-m3)

# 모델 이름
LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-m3"

# BGE-M3 임베딩 차원 (기본값: 1024)
EMBEDDING_DIM = 1024

# 작업 디렉토리
WORKING_DIR = "./vllm_test"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """
    vLLM을 통해 배포된 LLM 모델을 호출하는 함수
    """
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="EMPTY",  # vLLM은 API key가 필요하지 않지만 필드는 필요
        base_url=VLLM_LLM_HOST,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> list[list[float]]:
    """
    vLLM을 통해 배포된 임베딩 모델을 호출하는 함수
    """
    return await openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key="EMPTY",  # vLLM은 API key가 필요하지 않지만 필드는 필요
        base_url=VLLM_EMBEDDING_HOST,
    )


async def print_stream(stream):
    """스트리밍 응답 출력"""
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)
    print()  # 마지막에 줄바꿈


async def initialize_rag():
    """LightRAG 인스턴스 초기화"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    return rag


async def main():
    rag = None
    try:
        print("=" * 60)
        print("LightRAG + vLLM Demo")
        print("=" * 60)
        print(f"LLM Server: {VLLM_LLM_HOST}")
        print(f"LLM Model: {LLM_MODEL}")
        print(f"Embedding Server: {VLLM_EMBEDDING_HOST}")
        print(f"Embedding Model: {EMBEDDING_MODEL}")
        print(f"Embedding Dimension: {EMBEDDING_DIM}")
        print("=" * 60)

        # RAG 인스턴스 초기화
        print("\n[1/4] Initializing RAG instance...")
        rag = await initialize_rag()
        print("✓ RAG instance initialized")

        # 임베딩 함수 테스트
        print("\n[2/4] Testing embedding function...")
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print(f"✓ Test text: {test_text[0]}")
        print(f"✓ Detected embedding dimension: {embedding_dim}")

        # 샘플 문서 삽입
        print("\n[3/4] Inserting sample document...")
        sample_text = """
        LightRAG is a simple and fast Retrieval-Augmented Generation system.
        It uses graph-based knowledge representation to enhance query performance.
        The system supports multiple query modes: naive, local, global, hybrid, and mix.
        LightRAG can work with various storage backends including Neo4j, MongoDB, and PostgreSQL.
        """
        await rag.ainsert(sample_text)
        print("✓ Document inserted successfully")

        # 쿼리 테스트
        print("\n[4/4] Testing queries...")

        queries = [
            ("What is LightRAG?", "naive"),
            ("What are the query modes in LightRAG?", "local"),
            ("Tell me about LightRAG's features", "hybrid"),
        ]

        for i, (question, mode) in enumerate(queries, 1):
            print(f"\n{'─' * 60}")
            print(f"Query {i}/{len(queries)}")
            print(f"Mode: {mode}")
            print(f"Question: {question}")
            print(f"{'─' * 60}")

            resp = await rag.aquery(
                question,
                param=QueryParam(mode=mode, stream=True),
            )

            print("Answer: ", end="")
            if inspect.isasyncgen(resp):
                await print_stream(resp)
            else:
                print(resp)

        print("\n" + "=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
    print("\nDone!")
