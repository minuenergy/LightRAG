#!/usr/bin/env python3
"""
vLLM Quick Test Script
======================

vLLM 서버가 정상적으로 작동하는지 빠르게 테스트하는 스크립트입니다.

Usage:
    python examples/vllm_quick_test.py
"""

import asyncio
import sys
import requests
from typing import Optional


class Colors:
    """터미널 색상"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """헤더 출력"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def print_success(text: str):
    """성공 메시지 출력"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """에러 메시지 출력"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """경고 메시지 출력"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """정보 메시지 출력"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def test_server_connection(host: str, port: int, name: str) -> bool:
    """서버 연결 테스트"""
    url = f"http://{host}:{port}/v1/models"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json()
            print_success(f"{name} 서버 연결 성공 ({host}:{port})")
            if "data" in models and len(models["data"]) > 0:
                model_id = models["data"][0].get("id", "unknown")
                print_info(f"   사용 가능한 모델: {model_id}")
            return True
        else:
            print_error(f"{name} 서버 응답 오류 (상태 코드: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"{name} 서버에 연결할 수 없습니다 ({host}:{port})")
        print_info(f"   서버가 실행 중인지 확인하세요")
        return False
    except requests.exceptions.Timeout:
        print_error(f"{name} 서버 응답 시간 초과")
        return False
    except Exception as e:
        print_error(f"{name} 서버 테스트 중 오류: {e}")
        return False


def test_llm_completion(host: str, port: int) -> bool:
    """LLM 완성 테스트"""
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": "openai/gpt-oss-20b",
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            text = result.get("choices", [{}])[0].get("text", "")
            print_success(f"LLM 완성 테스트 성공")
            print_info(f"   응답: {text[:100]}..." if len(text) > 100 else f"   응답: {text}")
            return True
        else:
            print_error(f"LLM 완성 테스트 실패 (상태 코드: {response.status_code})")
            return False
    except Exception as e:
        print_error(f"LLM 완성 테스트 중 오류: {e}")
        return False


def test_embedding(host: str, port: int) -> Optional[int]:
    """임베딩 테스트"""
    url = f"http://{host}:{port}/v1/embeddings"
    payload = {
        "model": "BAAI/bge-m3",
        "input": "This is a test sentence for embedding."
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("data", [{}])[0].get("embedding", [])
            dim = len(embedding)
            print_success(f"임베딩 테스트 성공")
            print_info(f"   임베딩 차원: {dim}")
            return dim
        else:
            print_error(f"임베딩 테스트 실패 (상태 코드: {response.status_code})")
            return None
    except Exception as e:
        print_error(f"임베딩 테스트 중 오류: {e}")
        return None


async def test_lightrag_integration(embedding_dim: int):
    """LightRAG 통합 테스트"""
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
        import tempfile
        import shutil

        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()

        try:
            async def llm_func(prompt, system_prompt=None, **kwargs):
                return await openai_complete_if_cache(
                    "openai/gpt-oss-20b",
                    prompt,
                    system_prompt=system_prompt,
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1",
                    **kwargs,
                )

            async def embed_func(texts):
                return await openai_embed(
                    texts,
                    model="BAAI/bge-m3",
                    api_key="EMPTY",
                    base_url="http://localhost:8001/v1",
                )

            rag = LightRAG(
                working_dir=temp_dir,
                llm_model_func=llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dim,
                    max_token_size=8192,
                    func=embed_func,
                ),
            )

            await rag.initialize_storages()

            # 샘플 텍스트 삽입
            sample_text = "LightRAG is a graph-based RAG system."
            await rag.ainsert(sample_text)
            print_success("문서 삽입 테스트 성공")

            # 쿼리 테스트
            result = await rag.aquery(
                "What is LightRAG?",
                param=QueryParam(mode="naive")
            )
            print_success("쿼리 테스트 성공")
            print_info(f"   쿼리 결과: {str(result)[:100]}...")

            await rag.finalize_storages()
            return True

        finally:
            # 임시 디렉토리 삭제
            shutil.rmtree(temp_dir, ignore_errors=True)

    except ImportError as e:
        print_error(f"LightRAG 라이브러리를 가져올 수 없습니다: {e}")
        print_info("   pip install lightrag-hku 명령으로 설치하세요")
        return False
    except Exception as e:
        print_error(f"LightRAG 통합 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 함수"""
    print_header("vLLM + LightRAG 통합 테스트")

    # 설정
    LLM_HOST = "localhost"
    LLM_PORT = 8000
    EMBEDDING_HOST = "localhost"
    EMBEDDING_PORT = 8001

    all_passed = True

    # 1. LLM 서버 연결 테스트
    print_info("1/5: LLM 서버 연결 테스트")
    if not test_server_connection(LLM_HOST, LLM_PORT, "LLM"):
        all_passed = False
        print_warning("LLM 서버를 먼저 실행하세요:")
        print_info("   vllm serve openai/gpt-oss-20b --port 8000 \\")
        print_info("     --tensor-parallel-size 4 --gpu-memory-utilization 0.30 \\")
        print_info("     --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 8")

    # 2. Embedding 서버 연결 테스트
    print_info("\n2/5: Embedding 서버 연결 테스트")
    if not test_server_connection(EMBEDDING_HOST, EMBEDDING_PORT, "Embedding"):
        all_passed = False
        print_warning("Embedding 서버를 먼저 실행하세요:")
        print_info("   vllm serve BAAI/bge-m3 --port 8001 --gpu-memory-utilization 0.1")

    if not all_passed:
        print_error("\n서버 연결 실패. 나머지 테스트를 건너뜁니다.")
        sys.exit(1)

    # 3. LLM 완성 테스트
    print_info("\n3/5: LLM 완성 테스트")
    if not test_llm_completion(LLM_HOST, LLM_PORT):
        all_passed = False

    # 4. 임베딩 테스트
    print_info("\n4/5: Embedding 테스트")
    embedding_dim = test_embedding(EMBEDDING_HOST, EMBEDDING_PORT)
    if embedding_dim is None:
        all_passed = False

    # 5. LightRAG 통합 테스트
    print_info("\n5/5: LightRAG 통합 테스트")
    if embedding_dim and all_passed:
        if not asyncio.run(test_lightrag_integration(embedding_dim)):
            all_passed = False
    else:
        print_warning("이전 테스트 실패로 건너뜁니다")
        all_passed = False

    # 결과 요약
    print_header("테스트 결과")
    if all_passed:
        print_success("모든 테스트 통과! ✓")
        print_info("\n다음 단계:")
        print_info("  1. examples/lightrag_vllm_demo.py 실행")
        print_info("  2. 또는 .env 파일 설정 후 lightrag-server 실행")
        sys.exit(0)
    else:
        print_error("일부 테스트 실패")
        print_info("\n문제 해결 방법:")
        print_info("  1. vLLM 서버가 실행 중인지 확인")
        print_info("  2. 포트 8000, 8001이 사용 가능한지 확인")
        print_info("  3. GPU 메모리가 충분한지 확인")
        sys.exit(1)


if __name__ == "__main__":
    main()
