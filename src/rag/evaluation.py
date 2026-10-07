"""
RAG Evaluation and Benchmarking Suite for MediNexus AI.
Evaluates:
1. Retrieval Accuracy: Hit Rate @ 1, Hit Rate @ 3, Mean Reciprocal Rank (MRR), Context Relevance.
2. Latency Metrics: Vectorization / Retrieval Latency (ms), Synthesis Latency (ms), Total Latency (ms).
3. Grounding & Faithfulness: Citation verification against institutional policy documents.
"""

import time
import statistics
from typing import Dict, Any, List, Optional
from src.rag.retriever import retrieve_context
from src.rag.knowledge_base import generate_grounded_answer


# Curated Golden Benchmark Dataset: Clinical & Operational Queries with Ground-Truth Reference Documents
RAG_BENCHMARK_DATASET = [
    {
        "query_id": "Q1_CHF_DISCHARGE",
        "query": "What are the discharge criteria and self-management guidelines for chronic heart failure patients?",
        "expected_doc": "readmission_followup_guideline.txt",
        "category": "Clinical Guidelines",
        "expected_keywords": ["euvolemic", "diuretics", "body weight", "2 lbs"],
    },
    {
        "query_id": "Q2_HIGH_RISK_DISCHARGE",
        "query": "What is the policy protocol for inpatient discharge planning in high readmission risk patients?",
        "expected_doc": "discharge_policy.txt",
        "category": "Hospital Policies",
        "expected_keywords": ["Enhanced Transitional Protocol", "48 hours", "7 days"],
    },
    {
        "query_id": "Q3_APPOINTMENT_POLICY",
        "query": "What are the rules and guidelines for patient appointment scheduling, late arrivals, and cancellations?",
        "expected_doc": "appointment_policy.txt",
        "category": "Hospital Policies",
        "expected_keywords": ["appointment", "cancellation", "scheduling"],
    },
    {
        "query_id": "Q4_PATIENT_PRIVACY",
        "query": "What are the HIPAA patient privacy safeguards and minimum necessary access standards?",
        "expected_doc": "patient_privacy_policy.txt",
        "category": "Hospital Policies",
        "expected_keywords": ["privacy", "confidentiality", "minimum necessary"],
    },
    {
        "query_id": "Q5_LAB_SAFETY",
        "query": "What personal protective equipment and chemical biosafety procedures must laboratory staff follow?",
        "expected_doc": "laboratory_safety_policy.txt",
        "category": "Laboratory",
        "expected_keywords": ["protective equipment", "biohazard", "safety"],
    },
    {
        "query_id": "Q6_SAMPLE_HANDLING",
        "query": "What is the standard procedure for specimen chain of custody and critical panic values callback notification?",
        "expected_doc": "sample_handling_policy.txt",
        "category": "Laboratory",
        "expected_keywords": ["specimen", "chain of custody", "sample"],
    },
    {
        "query_id": "Q7_PHARMACY_STOCKOUT",
        "query": "What is the pharmacy replenishment and critical shortage protocol when medication stock drops below reorder point?",
        "expected_doc": "inventory_policy.txt",
        "category": "Pharmacy",
        "expected_keywords": ["inventory", "reorder point", "critical shortage", "formulary"],
    },
    {
        "query_id": "Q8_MEDICATION_STORAGE",
        "query": "What are the storage temperature requirements for refrigerated medications and cold chain integrity?",
        "expected_doc": "medication_storage_policy.txt",
        "category": "Pharmacy",
        "expected_keywords": ["storage", "refrigerated", "temperature", "cold chain"],
    },
]


def evaluate_retrieval_query(item: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    """
    Evaluate retrieval precision, hit rank, and latency for a single benchmark query.
    """
    query = item["query"]
    expected_doc = item["expected_doc"].lower()

    # Measure retrieval latency
    t0 = time.perf_counter()
    hits = retrieve_context(query, top_k=top_k)
    t_retrieval = time.perf_counter() - t0
    retrieval_ms = round(t_retrieval * 1000, 3)

    # Check Hit@K and Rank
    hit_at_1 = False
    hit_at_k = False
    reciprocal_rank = 0.0
    matched_rank = None
    top_score = 0.0

    for idx, hit in enumerate(hits):
        hit_fn = hit.get("filename", "").lower()
        if idx == 0:
            top_score = hit.get("score", 0.0)
            if expected_doc in hit_fn:
                hit_at_1 = True

        if expected_doc in hit_fn:
            hit_at_k = True
            matched_rank = idx + 1
            reciprocal_rank = 1.0 / (idx + 1)
            break

    # Keyword coverage in retrieved passages
    retrieved_text = " ".join([h.get("text", "") for h in hits]).lower()
    kw_matches = sum(1 for kw in item.get("expected_keywords", []) if kw.lower() in retrieved_text)
    kw_coverage = kw_matches / len(item.get("expected_keywords", [1])) if item.get("expected_keywords") else 1.0

    return {
        "query_id": item["query_id"],
        "query": query,
        "expected_doc": item["expected_doc"],
        "retrieval_latency_ms": retrieval_ms,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "matched_rank": matched_rank,
        "reciprocal_rank": reciprocal_rank,
        "top_similarity_score": round(top_score, 3),
        "keyword_coverage": round(kw_coverage * 100, 1),
        "retrieved_count": len(hits),
        "top_doc": hits[0]["filename"] if hits else "None",
    }


def evaluate_end_to_end_rag(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate end-to-end RAG answer synthesis latency, citation accuracy, and grounding.
    """
    query = item["query"]
    expected_doc = item["expected_doc"].lower()

    t0 = time.perf_counter()
    res = generate_grounded_answer(query)
    total_time = time.perf_counter() - t0
    total_ms = round(total_time * 1000, 3)

    citations = [c.get("filename", "").lower() for c in res.get("citations", [])]
    citation_match = any(expected_doc in c for c in citations)

    return {
        "query_id": item["query_id"],
        "total_latency_ms": total_ms,
        "citation_match": citation_match,
        "engine": res.get("engine", "Local"),
        "citations_count": len(res.get("citations", [])),
    }


def run_rag_benchmark(benchmark_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Execute complete RAG accuracy and latency benchmark across the test dataset.
    Returns aggregated metrics and per-query telemetry.
    """
    dataset = benchmark_items or RAG_BENCHMARK_DATASET

    retrieval_results = []
    e2e_results = []

    for item in dataset:
        ret_eval = evaluate_retrieval_query(item, top_k=3)
        retrieval_results.append(ret_eval)

        e2e_eval = evaluate_end_to_end_rag(item)
        e2e_results.append(e2e_eval)

    # Compute aggregate metrics
    total_queries = len(dataset)
    hits_at_1 = sum(1 for r in retrieval_results if r["hit_at_1"])
    hits_at_3 = sum(1 for r in retrieval_results if r["hit_at_k"])
    mrr = sum(r["reciprocal_rank"] for r in retrieval_results) / total_queries if total_queries else 0.0

    ret_latencies = [r["retrieval_latency_ms"] for r in retrieval_results]
    total_latencies = [e["total_latency_ms"] for e in e2e_results]
    citation_matches = sum(1 for e in e2e_results if e["citation_match"])

    avg_ret_lat = round(statistics.mean(ret_latencies), 2)
    p50_ret_lat = round(statistics.median(ret_latencies), 2)
    max_ret_lat = round(max(ret_latencies), 2)

    avg_tot_lat = round(statistics.mean(total_latencies), 2)
    p50_tot_lat = round(statistics.median(total_latencies), 2)
    max_tot_lat = round(max(total_latencies), 2)

    avg_kw_coverage = round(statistics.mean([r["keyword_coverage"] for r in retrieval_results]), 1)
    avg_similarity = round(statistics.mean([r["top_similarity_score"] for r in retrieval_results]), 3)

    summary = {
        "total_test_queries": total_queries,
        # Accuracy & Relevance
        "hit_rate_at_1_percent": round((hits_at_1 / total_queries) * 100, 1),
        "hit_rate_at_3_percent": round((hits_at_3 / total_queries) * 100, 1),
        "mrr_score": round(mrr, 4),
        "citation_accuracy_percent": round((citation_matches / total_queries) * 100, 1),
        "average_keyword_coverage_percent": avg_kw_coverage,
        "average_similarity_score": avg_similarity,
        # Latency Metrics (ms)
        "avg_retrieval_latency_ms": avg_ret_lat,
        "p50_retrieval_latency_ms": p50_ret_lat,
        "max_retrieval_latency_ms": max_ret_lat,
        "avg_total_latency_ms": avg_tot_lat,
        "p50_total_latency_ms": p50_tot_lat,
        "max_total_latency_ms": max_tot_lat,
        # Per query details
        "query_details": [
            {
                **retrieval_results[i],
                "total_latency_ms": e2e_results[i]["total_latency_ms"],
                "citation_match": e2e_results[i]["citation_match"],
            }
            for i in range(total_queries)
        ],
    }

    return summary


def print_benchmark_report(summary: Dict[str, Any]):
    """Print high-contrast terminal formatted benchmark report."""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 80)
    print("  MEDINEXUS AI -- RAG ACCURACY & LATENCY BENCHMARK REPORT")
    print("=" * 80)
    print(f"Total Test Queries Evaluated: {summary['total_test_queries']}")
    print("-" * 80)
    print("ACCURACY & RETRIEVAL METRICS:")
    print(f"  * Hit Rate @ 1:                  {summary['hit_rate_at_1_percent']}%")
    print(f"  * Hit Rate @ 3:                  {summary['hit_rate_at_3_percent']}%")
    print(f"  * Mean Reciprocal Rank (MRR@3):  {summary['mrr_score']:.4f}")
    print(f"  * Citation Accuracy:             {summary['citation_accuracy_percent']}%")
    print(f"  * Average Keyword Coverage:      {summary['average_keyword_coverage_percent']}%")
    print(f"  * Mean Cosine Relevance:         {summary['average_similarity_score']:.3f}")
    print("-" * 80)
    print("LATENCY & PERFORMANCE METRICS (ms):")
    print(f"  * Avg Retrieval Latency:         {summary['avg_retrieval_latency_ms']} ms  (p50: {summary['p50_retrieval_latency_ms']} ms, Max: {summary['max_retrieval_latency_ms']} ms)")
    print(f"  * Avg End-to-End Latency:        {summary['avg_total_latency_ms']} ms  (p50: {summary['p50_total_latency_ms']} ms, Max: {summary['max_total_latency_ms']} ms)")
    print("-" * 80)
    print(f"{'Query ID':<22} | {'Hit@1':<5} | {'Hit@3':<5} | {'MRR':<5} | {'Ret.(ms)':<9} | {'Tot.(ms)':<9} | {'Top Document':<26}")
    print("-" * 80)
    for q in summary["query_details"]:
        h1 = "YES" if q["hit_at_1"] else "NO"
        h3 = "YES" if q["hit_at_k"] else "NO"
        print(f"{q['query_id']:<22} | {h1:<5} | {h3:<5} | {q['reciprocal_rank']:<5.2f} | {q['retrieval_latency_ms']:<9.2f} | {q['total_latency_ms']:<9.2f} | {q['top_doc'][:26]:<26}")
    print("=" * 80)


if __name__ == "__main__":
    benchmark = run_rag_benchmark()
    print_benchmark_report(benchmark)
