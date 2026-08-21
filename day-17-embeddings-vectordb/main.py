import sys
from src.benchmark import run_evaluation_pipeline

def main():
    print("Executing Day 17: Embeddings, Vector Stores & Retrieval Benchmark...")
    results = run_evaluation_pipeline()
    print("\n[SUCCESS] Day 17 evaluation pipeline executed and outputs saved to outputs/benchmark_results.json")

if __name__ == "__main__":
    main()
