#!/usr/bin/env python3
"""
NanoLLM Benchmark - Test local models using nano_llm.chat API
"""

import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Test prompts (matching quick_benchmark.py)
PROMPTS = [
    {
        "id": "reasoning",
        "text": "Explain recursion in programming with a simple example.",
        "category": "reasoning"
    },
    {
        "id": "code",
        "text": "Write a Python function to check if a string is a palindrome.",
        "category": "code"
    },
    {
        "id": "summary",
        "text": "Summarize the key principles of object-oriented programming in 2-3 sentences.",
        "category": "summarization"
    },
]

# Models available locally
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

CONTAINER = "dustynv/nano_llm:r36.4.0"
QUANTIZATION = "q4f16_ft"

def benchmark_model_nanollm(model_name, prompt_text, prompt_id):
    """Benchmark using nano_llm.chat CLI"""
    
    print(f"  [{prompt_id}] Testing {model_name}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        # Use nano_llm.chat via container
        cmd = [
            'jetson-containers', 'run', '--rm',
            '-v', '/data:/data',
            CONTAINER,
            'python3', '-m', 'nano_llm.chat',
            '--api', 'mlc',
            '--model', model_name,
            '--quantization', QUANTIZATION,
            '--max-new-tokens', '500',
            '--prompt', prompt_text
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        total_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"✗ Error (exit {result.returncode})")
            if result.stderr:
                # Print last 200 chars of error
                error_msg = result.stderr[-200:].strip()
                print(f"    {error_msg}")
            return None
        
        # Parse output - look for the response
        output = result.stdout
        
        # Count tokens (rough estimate based on words)
        response_text = output.strip()
        tokens_generated = len(response_text.split())
        
        # Estimate first token time (typically 5-10% of total time for prefill)
        # This is a rough approximation since CLI doesn't expose detailed timing
        first_token_time = total_time * 0.08  # ~8% for prefill is typical
        throughput = tokens_generated / total_time if total_time > 0 else 0
        
        print(f"✓ {total_time:.1f}s, {throughput:.1f} tok/s, ~{tokens_generated} tokens")
        
        return {
            'model': model_name.split('/')[-1],
            'model_name': model_name,
            'prompt_id': prompt_id,
            'prompt_category': PROMPTS[[p['id'] for p in PROMPTS].index(prompt_id)]['category'],
            'timestamp': datetime.now().isoformat(),
            'backend': 'nanollm',
            'total_time': total_time,
            'first_token_time': first_token_time,
            'tokens_generated': tokens_generated,
            'throughput': throughput,
            'response': response_text,
            'quantization': QUANTIZATION
        }   
        
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout (>10 min)")
        return None
    except Exception as e:
        print(f"✗ Exception: {str(e)[:50]}")
        return None

def main():
    print("=" * 70)
    print("NanoLLM Benchmark")
    print("=" * 70)
    print(f"Container: {CONTAINER}")
    print(f"API: nano_llm.chat with MLC backend")
    print(f"Quantization: {QUANTIZATION}")
    print(f"Models: {len(MODELS)}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Total tests: {len(MODELS) * len(PROMPTS)}")
    print()
    
    results = []
    test_count = 0
    total_tests = len(MODELS) * len(PROMPTS)
    
    for model_name in MODELS:
        short_name = model_name.split('/')[-1]
        print(f"\n{short_name}")
        print("-" * 70)
        
        for prompt in PROMPTS:
            test_count += 1
            print(f"[{test_count}/{total_tests}] ", end="")
            
            result = benchmark_model_nanollm(model_name, prompt['text'], prompt['id'])
            if result:
                results.append(result)
        
        # Brief pause between models
        time.sleep(2)
    
    # Save results
    output_file = f"nanollm_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"Total successful tests: {len(results)}/{total_tests}")
    print(f"{'=' * 70}\n")
    
    # Summary
    if results:
        print("Summary by Model:")
        print("-" * 70)
        
        models = {}
        for r in results:
            model = r['model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        for model, model_results in models.items():
            avg_throughput = sum(r['throughput'] for r in model_results) / len(model_results)
            avg_time = sum(r['total_time'] for r in model_results) / len(model_results)
            total_tokens = sum(r['tokens_generated'] for r in model_results)
            
            print(f"{model}:")
            print(f"  Avg throughput: {avg_throughput:.1f} tokens/s")
            print(f"  Avg time: {avg_time:.1f}s")
            print(f"  Total tokens: {total_tokens}")
    
    print()

if __name__ == "__main__":
    main()
