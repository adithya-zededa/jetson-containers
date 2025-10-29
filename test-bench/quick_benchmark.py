#!/usr/bin/env python3
"""
Quick LLM Benchmark - Streamlined version for Ollama only
Tests available models with basic prompts
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path

# Test prompts (shorter for quick testing)
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

def get_available_models():
    """Get list of available Ollama models"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json()['models']]
            # Filter to just llama models
            return [m for m in models if 'llama' in m.lower() and 'text' not in m]
        return []
    except:
        return []

def unload_model(model):
    """Unload a model from Ollama to free memory"""
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": "",
                "keep_alive": 0
            },
            timeout=10
        )
        print(f"  ✓ Unloaded {model} from memory")
    except:
        pass

def benchmark_model(model, prompt_text):
    """Run a single benchmark"""
    print(f"  Testing {model}...", end=" ", flush=True)
    
    start_time = time.time()
    first_token_time = None
    tokens_generated = 0
    response_text = ""
    
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt_text,
                "stream": True
            },
            stream=True,
            timeout=120
        )
        
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    response_text += data['response']
                    tokens_generated += 1
                    
                if data.get('done', False):
                    break
        
        total_time = time.time() - start_time
        throughput = tokens_generated / total_time if total_time > 0 else 0
        
        print(f"✓ {total_time:.1f}s, {throughput:.1f} tok/s, {tokens_generated} tokens")
        
        return {
            "model": model,
            "total_time": total_time,
            "first_token_time": first_token_time or 0,
            "tokens_generated": tokens_generated,
            "throughput": throughput,
            "response": response_text
        }
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return None

def main():
    print("=" * 60)
    print("Quick LLM Benchmark")
    print("=" * 60)
    
    # Get available models
    models = get_available_models()
    if not models:
        print("Error: No Ollama models found or server not running")
        return
    
    print(f"\nFound {len(models)} models: {', '.join(models)}")
    print(f"Running {len(PROMPTS)} prompts per model\n")
    
    results = []
    total_tests = len(models) * len(PROMPTS)
    current = 0
    
    for model in models:
        print(f"\n{model}")
        print("-" * 60)
        
        for prompt in PROMPTS:
            current += 1
            print(f"[{current}/{total_tests}] {prompt['id']}: ", end="")
            
            result = benchmark_model(model, prompt['text'])
            if result:
                result.update({
                    "prompt_id": prompt['id'],
                    "prompt_category": prompt['category'],
                    "timestamp": datetime.now().isoformat()
                })
                results.append(result)
        
        # Unload model after testing to free memory
        print(f"\n  Unloading model...")
        unload_model(model)
    
    # Save results
    output_file = f"quick_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_file}")
    print(f"Total successful tests: {len(results)}/{total_tests}")
    
    # Quick summary
    if results:
        print(f"\n{'Summary'}")
        print("-" * 60)
        for model in models:
            model_results = [r for r in results if r['model'] == model]
            if model_results:
                avg_throughput = sum(r['throughput'] for r in model_results) / len(model_results)
                avg_time = sum(r['total_time'] for r in model_results) / len(model_results)
                total_tokens = sum(r['tokens_generated'] for r in model_results)
                print(f"{model}:")
                print(f"  Avg throughput: {avg_throughput:.1f} tokens/s")
                print(f"  Avg time: {avg_time:.1f}s")
                print(f"  Total tokens: {total_tokens}")

if __name__ == "__main__":
    main()
