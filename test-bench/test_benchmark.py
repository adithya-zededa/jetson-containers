#!/usr/bin/env python3
"""
Quick test run of the benchmark with a single model and prompt
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the benchmark class
from benchmark_llm import LLMBenchmark

# Create a test version with limited scope
class TestBenchmark(LLMBenchmark):
    """Limited benchmark for testing"""
    
    # Override with just one model and one prompt
    MODELS = ["meta-llama/Llama-3.2-1B-Instruct"]
    
    PROMPTS = [
        {
            "id": "test_prompt",
            "category": "reasoning",
            "prompt": "Explain what a hash table is and why it's useful in 2-3 sentences."
        }
    ]

def main():
    print("\n" + "="*60)
    print("Quick Test Run - Single Model, Single Prompt")
    print("="*60 + "\n")
    
    benchmark = TestBenchmark()
    
    try:
        # Check servers
        status = benchmark.check_servers()
        print(f"Server Status:")
        print(f"  NanoLLM: {'✓ Running' if status['nanollm'] else '✗ Not available'}")
        print(f"  Ollama:  {'✓ Running' if status['ollama'] else '✗ Not available'}")
        
        if not any(status.values()):
            print("\nError: No servers running. Start NanoLLM or Ollama first.")
            return
        
        print("\nRunning test benchmark...")
        benchmark.run_benchmarks()
        
        print("\nSaving results...")
        benchmark.save_results()
        
        print("\nGenerating visualizations...")
        benchmark.analyze_and_visualize()
        
        print("\n✅ Test completed successfully!")
        print("\nCheck the following files:")
        print("  - benchmark_results.json")
        print("  - quality_scores.json")
        print("  - results_summary.csv")
        print("  - plots/*.png")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
