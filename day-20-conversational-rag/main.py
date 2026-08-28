import os
import json
from rich.console import Console
from rich.table import Table
from src.conversational_rag import ConversationalRAGPipeline

console = Console()


def run_multi_turn_benchmark():
    console.print("[bold cyan]Day 20: LangChain Multi-Turn Conversational RAG Benchmark[/bold cyan]\n")
    
    dialogues_path = "data/multi_turn_conversations.json"
    if not os.path.exists(dialogues_path):
        console.print("[red]Missing data/multi_turn_conversations.json[/red]")
        return

    with open(dialogues_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    pipeline = ConversationalRAGPipeline(use_compression=False)
    results = []

    for conv in conversations:
        c_id = conv["dialogue_id"]
        pipeline.clear_session(c_id)
        console.print(f"[bold yellow]▶ Processing Dialogue: {c_id}[/bold yellow]")
        
        dialogue_log = {"dialogue_id": c_id, "turns": []}

        for idx, turn in enumerate(conv["turns"]):
            query = turn["query"]
            res = pipeline.ask(session_id=c_id, query=query)
            
            console.print(f"  [cyan]Turn {idx+1}:[/cyan] {query}")
            console.print(f"  [dim]Standalone Reformulation:[/dim] {res['standalone_query']}")
            console.print(f"  [green]Answer Preview:[/green] {res['answer'][:120]}...")
            console.print(f"  [magenta]Citations:[/magenta] {res['citations']}\n")

            dialogue_log["turns"].append({
                "turn": idx + 1,
                "raw_query": query,
                "standalone_query": res["standalone_query"],
                "answer": res["answer"],
                "citations": res["citations"]
            })
        
        results.append(dialogue_log)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/conversational_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    console.print("[bold green]✓ Benchmark completed. Saved outputs to outputs/conversational_benchmark_results.json[/bold green]")


if __name__ == "__main__":
    run_multi_turn_benchmark()