import os
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.retriever import VectorRetriever
from src.llm_client import LLMClient
from src.pipeline import SimpleRAGPipeline

console = Console()

def load_documents():
    """Loads hierarchical chunks if present, falling back to sample chunks."""
    hierarchical_path = os.path.join("data", "chunks_hierarchical.json")
    sample_path = os.path.join("data", "sample_chunks.json")

    if os.path.exists(hierarchical_path):
        target_path = hierarchical_path
        dataset_name = "Full Multimodal Hierarchical Dataset (Day 16 Corpus)"
    elif os.path.exists(sample_path):
        target_path = sample_path
        dataset_name = "Sample Baseline Dataset"
    else:
        raise FileNotFoundError("No chunk data found in data/chunks_hierarchical.json or data/sample_chunks.json")

    with open(target_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    formatted = []
    for idx, c in enumerate(raw_chunks):
        formatted.append({
            "id": c.get("id") or c.get("chunk_id") or f"chunk_{idx:04d}",
            "text": c.get("text") or c.get("content", ""),
            "metadata": c.get("metadata", {
                "source": c.get("source", "Course Documentation.pdf"),
                "page_number": c.get("page_number", 1),
                "section_heading": c.get("section_heading", "General Reference")
            })
        })

    console.print(f"[green]✓ Loaded {len(formatted)} chunks from: [bold]{dataset_name}[/bold][/green]")
    return formatted

def bootstrap_pipeline():
    """Initializes VectorRetriever (BAAI/bge-large-en-v1.5), ChromaDB, and LLM Client."""
    console.print("[bold blue]Initializing Day 18 Multimodal Simple RAG Pipeline...[/bold blue]")
    
    chunks = load_documents()

    retriever = VectorRetriever(
        collection_name="day18_rag_bge", 
        model_name="BAAI/bge-large-en-v1.5",
        persist_dir="outputs/chroma_db"
    )
    with console.status("[bold green]Indexing 1024-d BGE vectors into persistent ChromaDB...[/bold green]"):
        retriever.ingest_chunks(chunks)

    llm = LLMClient()
    mode = "Google Gemini Live API" if not llm.use_mock else "Deterministic Mock Engine"
    console.print(f"[green]✓ LLM Provider: {mode}[/green]")

    return SimpleRAGPipeline(retriever=retriever, llm_client=llm)

def run_evaluation_benchmark(pipeline: SimpleRAGPipeline):
    """Executes evaluation suite across technical and multimodal queries."""
    console.print("\n[bold magenta]=======================================================[/bold magenta]")
    console.print("[bold magenta] RUNNING SYSTEMATIC RAG EVALUATION SUITE              [/bold magenta]")
    console.print("[bold magenta]=======================================================[/bold magenta]\n")

    eval_path = os.path.join("data", "eval_questions.json")
    if not os.path.exists(eval_path):
        os.makedirs("data", exist_ok=True)
        default_questions = [
            {"id": "q1", "type": "Direct Retrieval", "question": "What is the formula for Mean Squared Error (MSE)?"},
            {"id": "q2", "type": "Multimodal Figure Analysis", "question": "What diagrams and loss trajectories are illustrated in the figures?"},
            {"id": "q3", "type": "Architectural Trade-off", "question": "Why would an engineer select a 384-dimension embedding model for edge computing?"},
            {"id": "q4", "type": "Negative / Out-of-Domain", "question": "How do you make a classic Neapolitan pizza?"}
        ]
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(default_questions, f, indent=2)

    with open(eval_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    table = Table(title="Day 18 RAG Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Type", style="yellow", width=22)
    table.add_column("Question", width=35)
    table.add_column("Grounded Answer & Citations", style="white", width=45)
    table.add_column("Latency", style="green", width=12)

    eval_results = []
    for item in questions:
        res = pipeline.run(item["question"], top_k=3)
        total_lat = f"{res['metrics']['total_pipeline_latency_ms']:.1f} ms"
        table.add_row(item["type"], item["question"], res["answer"], total_lat)
        eval_results.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": res["answer"],
            "metrics": res["metrics"],
            "citations": res["citations"]
        })

    console.print(table)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    console.print("\n[bold green]✓ Benchmark results saved to outputs/eval_benchmark_results.json[/bold green]")

def run_interactive_cli(pipeline: SimpleRAGPipeline):
    """Interactive CLI for query exploration."""
    console.print(Panel.fit(
        "[bold cyan]Simple RAG Interactive CLI (BGE-Large + ChromaDB + Gemini)[/bold cyan]\n"
        "Ask questions across your text passages, tables, and visual diagram captions.\n"
        "Type [bold red]'exit'[/bold red] to quit.",
        border_style="cyan"
    ))

    while True:
        try:
            query = console.input("\n[bold yellow]Ask Question > [/bold yellow]").strip()
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[blue]Exiting CLI. Day 18 verified![/blue]")
                break
            if not query:
                continue

            with console.status("[bold green]Retrieving context & generating answer...[/bold green]"):
                res = pipeline.run(query, top_k=3)

            console.print(f"\n[bold green]Answer:[/bold green]\n{res['answer']}")
            console.print(f"\n[cyan]Metadata Citations:[/cyan] {', '.join(res['citations']) if res['citations'] else 'None'}")
            console.print(f"[dim]Retrieval: {res['metrics']['retrieval_latency_ms']}ms | Generation: {res['metrics']['generation_latency_ms']}ms | Total: {res['metrics']['total_pipeline_latency_ms']}ms[/dim]")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 18 Simple RAG Pipeline Runner")
    parser.add_argument("--eval", action="store_true", help="Run automated evaluation")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI")
    args = parser.parse_args()

    pipeline = bootstrap_pipeline()
    if pipeline:
        if args.eval:
            run_evaluation_benchmark(pipeline)
        elif args.cli:
            run_interactive_cli(pipeline)
        else:
            run_evaluation_benchmark(pipeline)
            run_interactive_cli(pipeline)