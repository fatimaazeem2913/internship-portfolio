import os
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.pipeline_advanced import AdvancedRAGPipeline

console = Console()

def load_dataset():
    """Loads corpus documents, preferring hierarchical multimodal structures."""
    corpus_path = os.path.join("data", "sample_corpus.json")
    hierarchical_path = os.path.join("data", "chunks_hierarchical.json")

    target_path = corpus_path if os.path.exists(corpus_path) else hierarchical_path
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Corpus file not found at {corpus_path} or {hierarchical_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    formatted = []
    for idx, c in enumerate(raw_chunks):
        formatted.append({
            "id": str(c.get("id") or c.get("chunk_id") or f"chunk_{idx:04d}"),
            "text": c.get("text") or c.get("content", ""),
            "metadata": c.get("metadata", {
                "source": c.get("source", "SupportcoursesM-DLearning.pdf"),
                "page_number": c.get("page_number", 1),
                "section_heading": c.get("section_heading", "General Reference")
            })
        })

    console.print(f"[green]✓ Loaded {len(formatted)} multi-tier chunks into Hybrid Corpus[/green]")
    return formatted

def ensure_eval_questions():
    """Generates the 20-question benchmark dataset if not already present."""
    eval_path = os.path.join("data", "eval_20_questions.json")
    if not os.path.exists(eval_path):
        os.makedirs("data", exist_ok=True)
        questions = [
            {"id": "q01", "type": "factual_lookup", "question": "What is the mathematical loss formula for Mean Squared Error (MSE)?"},
            {"id": "q02", "type": "multimodal_diagram", "question": "What is the anatomy of a biological neuron and its mapping to an ANN?"},
            {"id": "q03", "type": "terminology", "question": "What is linear regression and what hypothesis function does it use?"},
            {"id": "q04", "type": "out_of_domain", "question": "What is the recipe and baking temperature for traditional Neapolitan sourdough pizza?"},
            {"id": "q05", "type": "factual_lookup", "question": "What is the cost function for linear regression with parameters w1 and w0?"},
            {"id": "q06", "type": "multimodal_diagram", "question": "Explain the workflow diagram and loss curves shown in the course figures."},
            {"id": "q07", "type": "terminology", "question": "What does the dendrite represent in an artificial neural network?"},
            {"id": "q08", "type": "out_of_domain", "question": "What are the orbital characteristics of Jupiter's moon Europa?"},
            {"id": "q09", "type": "factual_lookup", "question": "What is the relationship between synapse and weights in an ANN?"},
            {"id": "q10", "type": "terminology", "question": "What role does the cell nucleus play in neural network modeling?"},
            {"id": "q11", "type": "multimodal_diagram", "question": "Describe the architecture presented in Figure 4.1."},
            {"id": "q12", "type": "factual_lookup", "question": "What does the axon signify in an artificial neural network?"},
            {"id": "q13", "type": "out_of_domain", "question": "How do you repair a punctured bicycle tire?"},
            {"id": "q14", "type": "terminology", "question": "Define supervised learning in the context of regression models."},
            {"id": "q15", "type": "factual_lookup", "question": "How is prediction error computed in statistical learning?"},
            {"id": "q16", "type": "multimodal_diagram", "question": "What mapping is provided in Table 4.1 regarding biological and artificial neurons?"},
            {"id": "q17", "type": "out_of_domain", "question": "What is the boiling point of liquid nitrogen at standard pressure?"},
            {"id": "q18", "type": "terminology", "question": "What is the difference between biological synapses and ANN weights?"},
            {"id": "q19", "type": "factual_lookup", "question": "Where is the Mean Squared Error formula referenced in the course documentation?"},
            {"id": "q20", "type": "out_of_domain", "question": "Who composed the Four Seasons violin concertos?"}
        ]
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2)

def run_20_question_benchmark(pipeline: AdvancedRAGPipeline):
    """Executes the systematic 20-question evaluation comparing 4 retrieval methods."""
    console.print("\n[bold magenta]======================================================================[/bold magenta]")
    console.print("[bold magenta] DAY 19: 20-QUESTION SYSTEMATIC BENCHMARK ACROSS 4 RETRIEVAL METHODS  [/bold magenta]")
    console.print("[bold magenta]======================================================================[/bold magenta]\n")

    ensure_eval_questions()
    eval_path = os.path.join("data", "eval_20_questions.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    table = Table(title="Day 19 Retrieval Method Comparison", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=6)
    table.add_column("Category", style="yellow", width=18)
    table.add_column("Question", width=32)
    table.add_column("Simple Dense", width=14)
    table.add_column("BM25 Sparse", width=14)
    table.add_column("Hybrid (RRF)", width=14)
    table.add_column("Hybrid+ReRank", style="bold green", width=15)

    benchmark_records = []

    for item in questions:
        q = item["question"]
        q_id = item["id"]
        cat = item["type"]

        res_dense = pipeline.run(q, mode="simple_dense", use_query_rewriting=False, top_k=3)
        res_bm25 = pipeline.run(q, mode="bm25_sparse", use_query_rewriting=False, top_k=3)
        res_hybrid = pipeline.run(q, mode="hybrid_rrf", use_query_rewriting=True, top_k=3)
        res_rerank = pipeline.run(q, mode="hybrid_rerank", use_query_rewriting=True, top_k=3)

        # Quality scoring heuristics
        def score_res(res):
            ans_lower = res["answer"].lower()
            if cat == "out_of_domain":
                is_refusal = any(phrase in ans_lower for phrase in [
                    "sufficient information", "not contain", "sorry", "no relevant context"
                ])
                return "[green]Refused (100%)[/green]" if is_refusal else "[red]Hallucinated[/red]"
            citations = res["citations"]
            return f"[green]Rank 1 ({len(citations)} cited)[/green]" if citations else "[red]Missed[/red]"

        s_dense = score_res(res_dense)
        s_bm25 = score_res(res_bm25)
        s_hybrid = score_res(res_hybrid)
        s_rerank = score_res(res_rerank)

        table.add_row(q_id[:6], cat, q, s_dense, s_bm25, s_hybrid, s_rerank)

        benchmark_records.append({
            "id": q_id,
            "category": cat,
            "question": q,
            "dense_latency_ms": res_dense["metrics"]["total_pipeline_latency_ms"],
            "bm25_latency_ms": res_bm25["metrics"]["total_pipeline_latency_ms"],
            "hybrid_latency_ms": res_hybrid["metrics"]["total_pipeline_latency_ms"],
            "rerank_latency_ms": res_rerank["metrics"]["total_pipeline_latency_ms"],
            "rerank_answer": res_rerank["answer"],
            "citations": res_rerank["citations"]
        })

    console.print(table)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_20_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    console.print("\n[bold green]✓ Full 20-Question Benchmark saved to outputs/eval_20_benchmark_results.json[/bold green]")

def run_interactive_cli(pipeline: AdvancedRAGPipeline):
    """Interactive multi-mode comparison CLI."""
    console.print(Panel.fit(
        "[bold cyan]Day 19: Hybrid Search & Advanced Retrieval CLI[/bold cyan]\n"
        "Select mode to compare: [yellow]simple_dense[/yellow] | [yellow]bm25_sparse[/yellow] | [yellow]hybrid_rrf[/yellow] | [bold green]hybrid_rerank[/bold green]\n"
        "Type [bold red]'exit'[/bold red] to quit.",
        border_style="cyan"
    ))

    current_mode = "hybrid_rerank"
    while True:
        try:
            user_input = console.input(f"\n[bold yellow]({current_mode}) Ask Question (or /mode) > [/bold yellow]").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if user_input.startswith("/mode"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] in ["simple_dense", "bm25_sparse", "hybrid_rrf", "hybrid_rerank"]:
                    current_mode = parts[1]
                    console.print(f"[green]Switched mode to: [bold]{current_mode}[/bold][/green]")
                else:
                    console.print("[red]Valid modes: simple_dense, bm25_sparse, hybrid_rrf, hybrid_rerank[/red]")
                continue
            if not user_input:
                continue

            with console.status(f"[bold green]Executing {current_mode} retrieval & generation...[/bold green]"):
                res = pipeline.run(user_input, mode=current_mode, top_k=3)

            console.print(f"\n[cyan]Optimized Search Query:[/cyan] {res['search_query']}")
            console.print(f"\n[bold green]Answer:[/bold green]\n{res['answer']}")
            console.print(f"\n[cyan]Metadata Citations:[/cyan] {', '.join(res['citations']) if res['citations'] else 'None'}")
            console.print(f"[dim]Retrieval: {res['metrics']['retrieval_latency_ms']}ms | Generation: {res['metrics']['generation_latency_ms']}ms | Total: {res['metrics']['total_pipeline_latency_ms']}ms[/dim]")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 19 Hybrid Search Runner")
    parser.add_argument("--eval", action="store_true", help="Run 20-question benchmark")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI")
    args = parser.parse_args()

    chunks = load_dataset()
    pipeline = AdvancedRAGPipeline(corpus_chunks=chunks)

    if args.eval:
        run_20_question_benchmark(pipeline)
    elif args.cli:
        run_interactive_cli(pipeline)
    else:
        run_20_question_benchmark(pipeline)
        run_interactive_cli(pipeline)