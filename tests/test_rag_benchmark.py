"""
Tests for RAG Accuracy and Latency Benchmarks in MediNexus AI.
"""

import pytest
from src.rag.evaluation import run_rag_benchmark, evaluate_retrieval_query, evaluate_end_to_end_rag


def test_rag_retrieval_accuracy_benchmarks():
    """Verify that RAG knowledge retrieval meets rigorous enterprise accuracy thresholds."""
    summary = run_rag_benchmark()

    # Accuracy benchmarks
    assert summary["hit_rate_at_1_percent"] >= 90.0, f"Hit@1 too low: {summary['hit_rate_at_1_percent']}%"
    assert summary["hit_rate_at_3_percent"] == 100.0, f"Hit@3 not 100%: {summary['hit_rate_at_3_percent']}%"
    assert summary["mrr_score"] >= 0.90, f"MRR too low: {summary['mrr_score']}"
    assert summary["citation_accuracy_percent"] == 100.0, f"Citation accuracy too low: {summary['citation_accuracy_percent']}%"


def test_rag_latency_benchmarks():
    """Verify that RAG retrieval and synthesis satisfy real-time latency SLAs (< 100ms local)."""
    summary = run_rag_benchmark()

    # Latency benchmarks
    assert summary["avg_retrieval_latency_ms"] < 50.0, f"Retrieval latency exceeded 50ms: {summary['avg_retrieval_latency_ms']}ms"
    assert summary["p50_retrieval_latency_ms"] < 25.0, f"p50 retrieval latency exceeded 25ms: {summary['p50_retrieval_latency_ms']}ms"
    assert summary["avg_total_latency_ms"] < 100.0, f"Total latency exceeded 100ms: {summary['avg_total_latency_ms']}ms"
