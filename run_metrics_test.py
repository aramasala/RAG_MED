
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Config must load before rag_med.evaluation (uses configs.settings)
import configs.settings  # noqa: F401

from rag_med.evaluation.metrics import (
    compare_two_answers,
    evaluate_answer_pair_llm_alignment,
    evaluate_answer_pair_ragas_extended,
)

OUTPUT_JSON = ROOT / "test_metrics_result.json"
OUTPUT_TXT = ROOT / "test_metrics_result.txt"


def main() -> None:
    sample_path = Path(os.environ.get("RUN_METRICS_TEST_JSON", ROOT / "test_qa_sample.json"))
    if not sample_path.is_absolute():
        sample_path = ROOT / sample_path
    if not sample_path.exists():
        print(f"Error: sample file not found: {sample_path}")
        sys.exit(1)

    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)

    if not samples:
        print("Error: test_qa_sample.json is empty")
        sys.exit(1)

    lines: list[str] = []
    results: list[dict] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("ETALON (reference) vs VALUEAI (candidate)")
    log("  RAGAS: faithfulness only (context=etalon, response=ValueAI)")
    log("  Text:  cosine_similarity, factual_correctness")
    log("=" * 60)
    log(f"Sample file: {sample_path}")
    log(f"Items: {len(samples)}")
    log()

    for i, item in enumerate(samples):
        question = item.get("question", "")
        etalon = item.get("answer", "")
        valueai = item.get("valueai_answer", "")

        sample_result = {
            "index": i + 1,
            "question": question,
            "comparison_metrics": None,
        }

        log(f"--- Sample {i + 1} ---")
        log(f"Question: {question[:80]}...")
        log()

        if etalon and valueai:
            all_metrics: dict = {}

            # RAGAS: faithfulness only (etalon = context, ValueAI = response)
            log("RAGAS (etalon = context, ValueAI = response)")
            ragas = evaluate_answer_pair_ragas_extended(
                question=question,
                response=valueai,
                retrieved_contexts=[etalon],
                reference_answer=None,
            )
            all_metrics["ragas_faithfulness"] = ragas.get("faithfulness")
            if ragas.get("error"):
                all_metrics["ragas_error"] = ragas["error"]
            log(f"  faithfulness:  {ragas.get('faithfulness')}  (ValueAI claims supported by context)")
            if ragas.get("error"):
                log(f"  error: {ragas['error'][:200]}...")
            log()

            # Text overlap (etalon vs ValueAI): cosine_similarity, factual_correctness
            log("Text overlap (etalon vs ValueAI)")
            text = compare_two_answers(reference=etalon, candidate=valueai)
            all_metrics["cosine_similarity"] = text.get("cosine_similarity")
            all_metrics["factual_correctness"] = text.get("factual_correctness")
            log(f"  cosine_similarity:   {text.get('cosine_similarity')}  (bag-of-words)")
            log(f"  factual_correctness: {text.get('factual_correctness')}  (RAGAS, ref vs candidate)")
            if text.get("factual_error"):
                log(f"  factual_error: {text['factual_error'][:150]}...")
            log()

            # LLM judge: alignment (1–10) RAG vs etalon
            log("LLM judge (RAG vs etalon, 1–10)")
            llm_judge = evaluate_answer_pair_llm_alignment(
                question=question,
                etalon_answer=etalon,
                rag_answer=valueai,
                context=item.get("chunk", ""),
            )
            all_metrics["llm_alignment_score"] = llm_judge.get("alignment_score")
            all_metrics["llm_alignment_comment"] = llm_judge.get("alignment_comment")
            if llm_judge.get("alignment_error"):
                all_metrics["llm_alignment_error"] = llm_judge["alignment_error"]
            log(
                f"  alignment_score:     {llm_judge.get('alignment_score')}  "
                "(1–10, RAG vs etalon, LLM judge)"
            )
            if llm_judge.get("alignment_error"):
                log(f"  alignment_error: {llm_judge['alignment_error'][:200]}...")
            log()

            sample_result["comparison_metrics"] = all_metrics
        else:
            log("Skip: need both 'answer' (etalon) and 'valueai_answer'")
            log()

        results.append(sample_result)

    log("=" * 60)
    log("Done.")

    # Save JSON (full result for reuse)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\nResults saved to: {OUTPUT_JSON}")

    # Save readable text
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Results saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
