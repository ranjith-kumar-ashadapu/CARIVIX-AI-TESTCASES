"""
Evaluation Report Generator
============================

Assembles the results from the vector database evaluator and the
performance evaluator into:

    1. Well-formatted console output
    2. A comprehensive Markdown report (evaluation_report.md)

The report includes:
    - Current system configuration
    - Vector database evaluation
    - Retrieval performance metrics
    - Latency analysis
    - Similarity search results
    - Optimization comparison table
    - Best configuration
    - Observations
    - Recommendations
    - Conclusion
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from rag.evaluation.utils import format_elapsed

logger = logging.getLogger("CARIVIX_AI")

class EvaluationReport:
    """
    Generates console and Markdown reports from evaluation results.

    Attributes:
        vector_db_summary: Summary dict from VectorDBEvaluator.
        performance_results: List of config records from PerformanceEvaluator.
        best_config: Best configuration record.
        system_config: Current system configuration details.
    """

    def __init__(
        self,
        vector_db_summary: Dict[str, Any],
        performance_results: List[Dict[str, Any]],
        best_config: Optional[Dict[str, Any]],
        system_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the EvaluationReport.

        Args:
            vector_db_summary: Summary from VectorDBEvaluator.run_evaluation().
            performance_results: Results from PerformanceEvaluator.run_sweep().
            best_config: Record of the best configuration.
            system_config: Dict describing the current pipeline settings.
        """
        self.vector_db_summary = vector_db_summary or {}
        self.performance_results = performance_results or []
        self.best_config = best_config

        if system_config is None:
            system_config = {
                "Document Loader": "Implemented (PDF/DOCX/TXT/CSV)",
                "Text Preprocessing": "Enabled",
                "Chunk Size": 500,
                "Chunk Overlap": 50,
                "Embedding Model": "sentence-transformers/all-MiniLM-L6-v2",
                "Embedding Dimension": 384,
                "Vector Database": "FAISS",
                "Retrieval Top-k": 5,
                "LLM Runtime": "Ollama",
                "Processing Device": "CPU",
            }
        self.system_config = system_config

    # ==================================================================
    # Markdown Report
    # ==================================================================

    def generate_markdown_report(self) -> str:
        """
        Build the complete Markdown evaluation report.

        Returns:
            A string containing the full report in Markdown format.
        """
        lines: List[str] = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("# CARIVIX AI - RAG Pipeline Evaluation & Optimization Report")
        lines.append("")
        lines.append(f"_Generated: {now}_")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.extend(self._md_system_config())
        lines.append("")

        lines.extend(self._md_vector_db_evaluation())
        lines.append("")

        lines.extend(self._md_retrieval_performance())
        lines.append("")

        lines.extend(self._md_latency_analysis())
        lines.append("")

        lines.extend(self._md_similarity_search_results())
        lines.append("")

        lines.extend(self._md_optimization_comparison())
        lines.append("")

        lines.extend(self._md_best_configuration())
        lines.append("")

        lines.extend(self._md_observations())
        lines.append("")

        lines.extend(self._md_recommendations())
        lines.append("")

        lines.extend(self._md_conclusion())

        return "\n".join(lines)

    def _md_system_config(self) -> List[str]:
        """Markdown section: current system configuration."""
        lines = ["## 1. Current System Configuration", ""]
        lines.append("| Component | Configuration |")
        lines.append("|---|---|")
        for key, value in self.system_config.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        return lines

    def _md_vector_db_evaluation(self) -> List[str]:
        """Markdown section: vector database integration evaluation."""
        lines = ["## 2. Vector Database Evaluation", ""]
        iv = self.vector_db_summary.get("index_valid", {})
        searches = self.vector_db_summary.get("searches", [])

        lines.append("### FAISS Index Validation")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|---|---|")
        lines.append(f"| FAISS Index Loaded | {iv.get('index_loaded', 'N/A')} |")
        lines.append(f"| Status | {iv.get('status', 'N/A')} |")
        lines.append(f"| Vector Count | {iv.get('vector_count', 0)} |")
        lines.append(f"| Document Count | {iv.get('document_count', 0)} |")
        lines.append(f"| Embedding Dimension | {iv.get('dimension', 'N/A')} |")
        lines.append(f"| Expected Dimension | {iv.get('expected_dimension', 'N/A')} |")
        lines.append(
            f"| Dimension Match | {iv.get('dimension_match', 'N/A')} |"
        )
        lines.append(
            f"| Index Load Time | {format_elapsed(iv.get('index_load_time', 0))} |"
        )
        lines.append(f"| Index Path | `{iv.get('index_path', 'N/A')}` |")
        lines.append(f"| Metadata Path | `{iv.get('metadata_path', 'N/A')}` |")
        lines.append("")

        if iv.get("errors"):
            lines.append("**Validation Errors:**")
            for err in iv["errors"]:
                lines.append(f"  - {err}")
            lines.append("")

        lines.append("### Similarity Search Summary")
        lines.append("")
        if searches:
            lines.append("| Query | Results | Avg Score | Lexical Rel. | Duplicates | Retrieval |")
            lines.append("|---|---|---|---|---|---|")
            for s in searches:
                lines.append(
                    f"| {s['query'][:60]} | {s['num_results']} | "
                    f"{s['avg_score']:.4f} | {s['lexical_relevance']:.4f} | "
                    f"{s['duplicates']} | {format_elapsed(s['retrieval_time'])} |"
                )
        else:
            lines.append("_No similarity searches were performed._")
        lines.append("")

        lines.append("### Detailed Retrieved Documents")
        lines.append("")
        if searches:
            lines.append("| Query | Rank | Chunk ID | Source | Score | Text Preview |")
            lines.append("|---|---|---|---|---|---|")
            for s in searches:
                for r in s["results"]:
                    text = r["document"].page_content.replace("\n", " ")[:60]
                    lines.append(
                        f"| {s['query'][:40]} | {r['rank']} | {r['chunk_id']} | "
                        f"{r['source']} | {r['score']:.4f} | {text}... |"
                    )
        else:
            lines.append("_No search results available._")
        lines.append("")

        overall = self.vector_db_summary.get("overall_status", "N/A")
        lines.append(f"**Overall Vector DB Status: `{overall}`**")
        lines.append("")
        return lines

    def _md_retrieval_performance(self) -> List[str]:
        """Markdown section: measured retrieval performance metrics."""
        lines = ["## 3. Retrieval Performance Metrics", ""]

        if not self.performance_results:
            lines.append("_No performance sweep was executed._")
            lines.append("")
            return lines

        lines.append("| Config | Chunks | Embed Time | Index Time | Retrieval | Total Query |")
        lines.append("|---|---|---|---|---|---|")
        for r in self.performance_results:
            lines.append(
                f"| **{r['chunk_size']}/{r['chunk_overlap']}/k={r['top_k']}** | "
                f"{r['chunk_count']} | {format_elapsed(r['embedding_time'])} | "
                f"{format_elapsed(r['indexing_time'])} | "
                f"{format_elapsed(r['retrieval_latency'])} | "
                f"{format_elapsed(r['total_query_time'])} |"
            )
        lines.append("")
        return lines

    def _md_latency_analysis(self) -> List[str]:
        """Markdown section: latency breakdown analysis."""
        lines = ["## 4. Latency Analysis", ""]

        rows = self.performance_results
        if not rows:
            lines.append("_No latency data available._")
            lines.append("")
            return lines

        # Compute averages across all configs.
        avg_sim = sum(r["avg_similarity"] for r in rows) / len(rows)
        avg_retrieval = sum(r["retrieval_latency"] for r in rows) / len(rows)
        avg_embed = sum(r["embedding_time"] for r in rows) / len(rows)
        avg_index = sum(r["indexing_time"] for r in rows) / len(rows)

        lines.append("### Average Metrics Across All Configurations")
        lines.append("")
        lines.append("| Metric | Average |")
        lines.append("|---|---|")
        lines.append(f"| Embedding Generation (batch) | {format_elapsed(avg_embed)} |")
        lines.append(f"| FAISS Indexing (batch)   | {format_elapsed(avg_index)} |")
        lines.append(f"| Retrieval Latency (per query) | {format_elapsed(avg_retrieval)} |")
        lines.append(f"| Average Similarity Score | {avg_sim:.4f} |")
        lines.append("")

        # Fastest & slowest configurations.
        fastest = min(rows, key=lambda r: r["retrieval_latency"])
        slowest = max(rows, key=lambda r: r["retrieval_latency"])
        lines.append("### Fastest vs Slowest Configuration")
        lines.append("")
        lines.append("| Metric | Fastest | Slowest |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Config | {fastest['chunk_size']}/{fastest['chunk_overlap']}/k={fastest['top_k']} | "
            f"{slowest['chunk_size']}/{slowest['chunk_overlap']}/k={slowest['top_k']} |"
        )
        lines.append(
            f"| Retrieval Latency | {format_elapsed(fastest['retrieval_latency'])} | "
            f"{format_elapsed(slowest['retrieval_latency'])} |"
        )
        lines.append(
            f"| Average Similarity | {fastest['avg_similarity']:.4f} | "
            f"{slowest['avg_similarity']:.4f} |"
        )
        lines.append("")
        return lines

    def _md_similarity_search_results(self) -> List[str]:
        """Markdown section: per-query similarity search results."""
        lines = ["## 5. Similarity Search Results", ""]
        searches = self.vector_db_summary.get("searches", [])
        if not searches:
            lines.append("_No similarity search data available._")
            lines.append("")
            return lines

        for s in searches:
            lines.append(f"### Query: {s['query']}")
            lines.append("")
            lines.append("| Rank | Chunk ID | Source | Score | Text |")
            lines.append("|---|---|---|---|---|")
            for r in s["results"]:
                text = r["document"].page_content.replace("\n", " ")[:80]
                lines.append(
                    f"| {r['rank']} | {r['chunk_id']} | {r['source']} | "
                    f"{r['score']:.4f} | {text}... |"
                )
            lines.append("")
        return lines

    def _md_optimization_comparison(self) -> List[str]:
        """Markdown section: optimization comparison table."""
        lines = ["## 6. Optimization Comparison Table", ""]
        rows = self.performance_results
        if not rows:
            lines.append("_No optimization data available._")
            lines.append("")
            return lines

        lines.append("| Rank | Chunk Size | Overlap | Top-k | Avg Similarity | Lexical Rel. | Retrieval | Total Query | Composite |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for rank, r in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {r['chunk_size']} | {r['chunk_overlap']} | {r['top_k']} | "
                f"{r['avg_similarity']:.4f} | {r['lexical_relevance']:.4f} | "
                f"{format_elapsed(r['retrieval_latency'])} | "
                f"{format_elapsed(r['total_query_time'])} | "
                f"{r['composite_score']:.4f} |"
            )
        lines.append("")
        return lines

    def _md_best_configuration(self) -> List[str]:
        """Markdown section: best configuration."""
        lines = ["## 7. Best Configuration", ""]
        b = self.best_config
        if not b:
            lines.append("_No best configuration available._")
            lines.append("")
            return lines

        lines.append("| Parameter | Recommended Value |")
        lines.append("|---|---|")
        lines.append(f"| Chunk Size | {b['chunk_size']} |")
        lines.append(f"| Chunk Overlap | {b['chunk_overlap']} |")
        lines.append(f"| Retrieval Top-k | {b['top_k']} |")
        lines.append(f"| Number of Chunks | {b['chunk_count']} |")
        lines.append(f"| Embedding Time | {format_elapsed(b['embedding_time'])} |")
        lines.append(f"| FAISS Indexing Time | {format_elapsed(b['indexing_time'])} |")
        lines.append(f"| Retrieval Latency | {format_elapsed(b['retrieval_latency'])} |")
        lines.append(f"| Total Query Time | {format_elapsed(b['total_query_time'])} |")
        lines.append(f"| Average Similarity | {b['avg_similarity']:.4f} |")
        lines.append(f"| Lexical Relevance | {b['lexical_relevance']:.4f} |")
        lines.append(f"| Composite Score | {b['composite_score']:.4f} |")
        lines.append("")
        return lines

    def _md_observations(self) -> List[str]:
        """Markdown section: observations and reasoning."""
        lines = ["## 8. Observations", ""]
        observations: List[str] = []
        b = self.best_config
        rows = self.performance_results

        if self.vector_db_summary.get("index_valid", {}).get("index_loaded"):
            iv = self.vector_db_summary["index_valid"]
            observations.append(
                f"The FAISS index loads successfully with {iv['vector_count']} "
                f"vectors at dimension {iv['dimension']}."
            )
        if self.vector_db_summary.get("total_duplicates", 0) > 0:
            observations.append(
                f"Duplicate chunk IDs were detected across "
                f"{self.vector_db_summary['total_duplicates']} retrievals."
            )
        else:
            observations.append(
                "No duplicate retrieval results were detected across sample queries."
            )

        if rows:
            fastest = min(rows, key=lambda r: r["retrieval_latency"])
            highest_sim = max(rows, key=lambda r: r["avg_similarity"])
            observations.append(
                f"The fastest retrieval config is {fastest['chunk_size']}/"
                f"{fastest['chunk_overlap']}/k={fastest['top_k']} "
                f"({format_elapsed(fastest['retrieval_latency'])}/query)."
            )
            observations.append(
                f"The highest-average-similarity config is "
                f"{highest_sim['chunk_size']}/{highest_sim['chunk_overlap']}/"
                f"k={highest_sim['top_k']} (avg similarity "
                f"{highest_sim['avg_similarity']:.4f})."
            )

        if b:
            # Compare best to a baseline (current 500/50/k=5) if it exists.
            baseline = next(
                (
                    r for r in rows
                    if r["chunk_size"] == 500
                    and r["chunk_overlap"] == 50
                    and r["top_k"] == 5
                ),
                None,
            )
            if baseline and baseline != b:
                delta_latency = (
                    b["retrieval_latency"] - baseline["retrieval_latency"]
                )
                delta_sim = b["avg_similarity"] - baseline["avg_similarity"]
                observations.append(
                    f"Compared to the current default (500/50/k=5), the best "
                    f"config changes retrieval latency by {delta_latency:+.4f}s "
                    f"and average similarity by {delta_sim:+.4f}."
                )

        for obs in observations:
            lines.append(f"- {obs}")
        lines.append("")
        return lines

    def _md_recommendations(self) -> List[str]:
        """Markdown section: recommendations."""
        lines = ["## 9. Recommendations", ""]
        b = self.best_config

        if b:
            lines.append(
                f"1. **Adopt the recommended configuration** "
                f"(chunk_size={b['chunk_size']}, chunk_overlap={b['chunk_overlap']}, "
                f"top_k={b['top_k']}) based on its composite score balancing "
                f"retrieval speed and relevance."
            )
        lines.append(
            "2. **Apply score-based filtering** (e.g., drop chunks below a "
            "similarity threshold) to reduce noise in the retrieved context."
        )
        lines.append(
            "3. **Consider re-indexing the production vector store** with the "
            "recommended chunking parameters for a measurable retrieval quality gain."
        )
        lines.append(
            "4. **Use batched embedding generation** and persist the "
            "sentence-transformer model locally to reduce cold-start latency."
        )
        lines.append(
            "5. **Periodically re-run this evaluation** (evaluate_rag.py) "
            "when documents or the embedding model change."
        )
        lines.append("")
        return lines

    def _md_conclusion(self) -> List[str]:
        """Markdown section: conclusion."""
        lines = ["## 10. Conclusion", ""]
        overall = self.vector_db_summary.get("overall_status", "N/A")
        b = self.best_config

        lines.append(
            f"The CARIVIX AI RAG pipeline's vector database integration is "
            f"**{overall}**."
        )
        lines.append("")
        if b:
            lines.append(
                f"Based on a grid search over chunk size, chunk overlap, and "
                f"top-k, the recommended configuration is "
                f"**chunk_size={b['chunk_size']}, "
                f"chunk_overlap={b['chunk_overlap']}, top_k={b['top_k']}** "
                f"with an average similarity of {b['avg_similarity']:.4f} and "
                f"a retrieval latency of {format_elapsed(b['retrieval_latency'])}."
            )
        lines.append("")
        lines.append(
            "These recommendations are intended as a starting point; adopt them "
            "incrementally and validate against real user queries."
        )
        lines.append("")
        return lines

    # ==================================================================
    # Console Report
    # ==================================================================

    def save_markdown_report(self, output_path: str = "evaluation_report.md") -> str:
        """
        Write the Markdown report to disk.

        Args:
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.
        """
        content = self.generate_markdown_report()

        directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(directory, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Markdown report saved to: %s", os.path.abspath(output_path))
        return os.path.abspath(output_path)

    def print_console_report(self) -> None:
        """Print a well-formatted summary report to the console."""
        width = 70
        print("\n" + "=" * width)
        print("  CARIVIX AI - RAG PIPELINE EVALUATION REPORT")
        print("=" * width)
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n" + "-" * width)
        print("  SYSTEM CONFIGURATION")
        print("-" * width)
        for key, value in self.system_config.items():
            print(f"  {key:<25} {value}")

        # Vector DB summary
        if self.vector_db_summary:
            iv = self.vector_db_summary.get("index_valid", {})
            print("\n" + "-" * width)
            print("  VECTOR DATABASE EVALUATION")
            print("-" * width)
            print(f"  Index Loaded:        {iv.get('index_loaded', 'N/A')}")
            print(f"  Index Status:        {iv.get('status', 'N/A')}")
            print(f"  Vector Count:        {iv.get('vector_count', 0)}")
            print(f"  Dimension:           {iv.get('dimension', 'N/A')}")
            print(f"  Dimension Match:     {iv.get('dimension_match', 'N/A')}")
            print(
                f"  Overall Status:      "
                f"{self.vector_db_summary.get('overall_status', 'N/A')}"
            )
            print(
                f"  Avg Similarity:      "
                f"{self.vector_db_summary.get('avg_score_overall', 0):.4f}"
            )
            print(
                f"  Total Duplicates:    "
                f"{self.vector_db_summary.get('total_duplicates', 0)}"
            )

        # Performance summary
        if self.performance_results:
            print("\n" + "-" * width)
            print("  PERFORMANCE OPTIMIZATION SWEEP")
            print("-" * width)
            print(
                f"  Configurations Evaluated: {len(self.performance_results)}"
            )
            if self.best_config:
                b = self.best_config
                print(
                    f"  Best Config:           chunk_size={b['chunk_size']}, "
                    f"overlap={b['chunk_overlap']}, top_k={b['top_k']}"
                )
                print(
                    f"  Best Retrieval Latency: "
                    f"{format_elapsed(b['retrieval_latency'])}"
                )
                print(
                    f"  Best Avg Similarity:   {b['avg_similarity']:.4f}"
                )
                print(
                    f"  Best Composite Score:  {b['composite_score']:.4f}"
                )

        print("\n" + "=" * width)
        print("  REPORT SAVED TO: evaluation_report.md")
        print("=" * width + "\n")

