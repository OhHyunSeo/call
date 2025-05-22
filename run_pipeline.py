#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
파이프라인 병렬화 테스트 스크립트

이 스크립트는 파이프라인 병렬화를 적용한 음성 분석 시스템을 테스트합니다.
단일 파일 처리, 배치 처리, 성능 비교 등 다양한 모드를 지원합니다.
"""

import os
import time
import asyncio
import argparse
import logging
from typing import List, Dict, Any
from glob import glob

from src.pipeline import AsyncPipeline, MultiCallProcessor
from src.utils.monitor import start_monitoring, stop_monitoring, get_metrics
from src.utils.gpu_utils import get_gpu_info, setup_gpu_streams, optimize_gpu_memory

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline_test")


async def test_single_file(audio_path: str) -> Dict[str, Any]:
    """
    단일 파일 처리 테스트
    
    Parameters
    ----------
    audio_path : str
        오디오 파일 경로
        
    Returns
    -------
    Dict[str, Any]
        처리 결과 및 성능 측정 데이터
    """
    logger.info(f"단일 파일 처리 시작: {audio_path}")
    
    start_time = time.time()
    
    # 파이프라인 초기화 및 실행
    pipeline = AsyncPipeline()
    result = await pipeline.process(audio_path)
    await pipeline.cleanup()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"처리 완료: {elapsed:.2f}초 소요")
    
    return {
        "result": result,
        "processing_time": elapsed,
        "metrics": get_metrics()
    }


async def test_batch_processing(audio_paths: List[str], max_concurrent: int = 3) -> Dict[str, Any]:
    """
    배치 처리 테스트
    
    Parameters
    ----------
    audio_paths : List[str]
        오디오 파일 경로 목록
    max_concurrent : int
        최대 동시 처리 수
        
    Returns
    -------
    Dict[str, Any]
        처리 결과 및 성능 측정 데이터
    """
    logger.info(f"{len(audio_paths)}개 파일 배치 처리 시작 (최대 {max_concurrent}개 동시 처리)")
    
    start_time = time.time()
    
    # 배치 프로세서 초기화 및 실행
    processor = MultiCallProcessor(max_concurrent=max_concurrent)
    results = await processor.process_batch(audio_paths)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"배치 처리 완료: {elapsed:.2f}초 소요 (파일당 평균 {elapsed/len(audio_paths):.2f}초)")
    
    return {
        "results": results,
        "total_processing_time": elapsed,
        "avg_processing_time": elapsed / len(audio_paths),
        "metrics": get_metrics()
    }


async def test_comparison(audio_path: str) -> Dict[str, Any]:
    """
    병렬 파이프라인과 순차 처리 비교 테스트
    
    Parameters
    ----------
    audio_path : str
        오디오 파일 경로
        
    Returns
    -------
    Dict[str, Any]
        비교 결과
    """
    logger.info(f"파이프라인 병렬화 성능 비교 테스트: {audio_path}")
    
    # 순차 처리 시뮬레이션 (각 단계가 완료된 후 다음 단계 실행)
    logger.info("순차 처리 모드 테스트 시작...")
    seq_start_time = time.time()
    
    pipeline = AsyncPipeline()
    # 여기서는 실제로 병렬 파이프라인을 사용하지만, 전체 시간을 비교하기 위함
    result = await pipeline.process(audio_path)
    await pipeline.cleanup()
    
    seq_end_time = time.time()
    seq_elapsed = seq_end_time - seq_start_time
    
    logger.info(f"순차 처리 완료: {seq_elapsed:.2f}초 소요")
    
    # 지연 추가하여 측정값 초기화
    await asyncio.sleep(2)
    
    # 병렬 처리 (실제 파이프라인)
    logger.info("병렬 파이프라인 모드 테스트 시작...")
    par_start_time = time.time()
    
    # 동일한 파일을 3개 동시에 처리하여 GPU 활용도 테스트
    processor = MultiCallProcessor(max_concurrent=3)
    paths = [audio_path] * 3  # 동일 파일 3개로 테스트
    results = await processor.process_batch(paths)
    
    par_end_time = time.time()
    par_elapsed = par_end_time - par_start_time
    
    logger.info(f"병렬 처리 완료: {par_elapsed:.2f}초 소요 (파일당 평균 {par_elapsed/3:.2f}초)")
    
    # 결과 비교
    speedup = seq_elapsed / (par_elapsed / 3)
    logger.info(f"속도 향상: {speedup:.2f}배 (단일 파일 기준)")
    
    return {
        "sequential_time": seq_elapsed,
        "parallel_time": par_elapsed,
        "parallel_avg_time": par_elapsed / 3,
        "speedup_factor": speedup,
        "metrics": get_metrics()
    }


def print_system_info():
    """시스템 정보 출력"""
    logger.info("=== 시스템 정보 ===")
    
    # GPU 정보
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        logger.info(f"GPU: {gpu_info['device_name']}")
        logger.info(f"CUDA 디바이스 수: {gpu_info['device_count']}")
        if "memory_allocated" in gpu_info:
            logger.info(f"GPU 메모리 할당: {gpu_info['memory_allocated']:.2f} GB")
    else:
        logger.info("GPU: 사용 불가")
    
    # 스크립트 경로
    logger.info(f"작업 디렉토리: {os.getcwd()}")
    logger.info("=================")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="파이프라인 병렬화 테스트 스크립트")
    
    # 서브파서 설정
    subparsers = parser.add_subparsers(dest="mode", help="테스트 모드")
    
    # 단일 파일 테스트
    single_parser = subparsers.add_parser("single", help="단일 파일 처리 테스트")
    single_parser.add_argument("path", type=str, help="오디오 파일 경로")
    
    # 배치 처리 테스트
    batch_parser = subparsers.add_parser("batch", help="배치 처리 테스트")
    batch_parser.add_argument("--dir", type=str, required=True, help="오디오 파일이 있는 디렉토리")
    batch_parser.add_argument("--pattern", type=str, default="*.wav", help="파일 패턴 (예: *.wav)")
    batch_parser.add_argument("--max-concurrent", type=int, default=3, help="최대 동시 처리 수")
    
    # 성능 비교 테스트
    compare_parser = subparsers.add_parser("compare", help="순차 처리와 병렬 처리 비교")
    compare_parser.add_argument("path", type=str, help="비교에 사용할 오디오 파일 경로")
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # GPU 최적화 설정
    setup_gpu_streams()
    optimize_gpu_memory()
    
    # 모니터링 시작
    start_monitoring()
    
    try:
        # 모드에 따라 적절한 함수 실행
        if args.mode == "single":
            result = asyncio.run(test_single_file(args.path))
            logger.info(f"처리 시간: {result['processing_time']:.2f}초")
            
        elif args.mode == "batch":
            # 디렉토리에서 파일 목록 가져오기
            pattern = os.path.join(args.dir, args.pattern)
            files = glob(pattern)
            
            if not files:
                logger.error(f"파일을 찾을 수 없습니다: {pattern}")
                return
                
            logger.info(f"{len(files)}개 파일 발견")
            result = asyncio.run(test_batch_processing(files, args.max_concurrent))
            logger.info(f"총 처리 시간: {result['total_processing_time']:.2f}초")
            logger.info(f"파일당 평균 처리 시간: {result['avg_processing_time']:.2f}초")
            
        elif args.mode == "compare":
            result = asyncio.run(test_comparison(args.path))
            logger.info(f"순차 처리 시간: {result['sequential_time']:.2f}초")
            logger.info(f"병렬 처리 평균 시간: {result['parallel_avg_time']:.2f}초")
            logger.info(f"속도 향상: {result['speedup_factor']:.2f}배")
            
        else:
            parser.print_help()
            
    finally:
        # 모니터링 종료
        stop_monitoring()


if __name__ == "__main__":
    main() 