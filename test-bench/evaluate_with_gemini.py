#!/usr/bin/env python3
"""
Evaluate benchmark results using Gemini as judge
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed. Run: pip3 install google-generativeai")
    sys.exit(1)

def evaluate_response(gemini_model, prompt, response):
    """Evaluate a single response using Gemini"""
    
    judge_prompt = f"""You are an expert AI evaluator. Evaluate the following LLM response on a scale of 1-10 for each criterion.

Prompt: {prompt}

Response: {response}

Evaluate and provide scores (1-10) for:
1. Correctness: Is the information accurate and factually correct?
2. Relevance: How well does it address the prompt?
3. Completeness: Is the answer thorough and complete?
4. Coherence: Is it well-structured and easy to follow?
5. Reasoning Quality: Is the logic sound and well-explained?

Respond ONLY with a JSON object in this exact format:
{{
  "correctness": <score>,
  "relevance": <score>,
  "completeness": <score>,
  "coherence": <score>,
  "reasoning_quality": <score>
}}"""
    
    try:
        response_obj = gemini_model.generate_content(judge_prompt)
        scores_text = response_obj.text.strip()
        
        # Extract JSON from response
        json_start = scores_text.find('{')
        json_end = scores_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            scores = json.loads(scores_text[json_start:json_end])
            avg = sum(scores.values()) / len(scores)
            return {**scores, 'average': avg}
        else:
            print(f"    Warning: No JSON found in response")
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Find benchmark results (both Ollama and NanoLLM)
    ollama_files = sorted(Path('.').glob('quick_benchmark_*.json'), reverse=True)
    nanollm_files = sorted(Path('.').glob('nanollm_benchmark_*.json'), reverse=True)
    
    all_files = []
    if ollama_files:
        all_files.append(('Ollama', ollama_files[0]))
    if nanollm_files:
        all_files.append(('NanoLLM', nanollm_files[0]))
    
    if not all_files:
        print("Error: No benchmark results found")
        print("Looking for: quick_benchmark_*.json or nanollm_benchmark_*.json")
        sys.exit(1)
    
    print("Found benchmark results:")
    for backend, filepath in all_files:
        print(f"  [{backend}] {filepath}")
    print()
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("✓ Gemini API configured (gemini-2.5-flash)\n")
    
    # Evaluate all results
    all_results = []
    all_evaluations = []
    
    for backend, filepath in all_files:
        print(f"\n{'='*70}")
        print(f"Evaluating {backend} results from: {filepath}")
        print(f"{'='*70}\n")
        
        # Load results
        with open(filepath) as f:
            results = json.load(f)
        
        print(f"Found {len(results)} results to evaluate\n")
        all_results.extend(results)
        
        # Evaluate each result
        for i, result in enumerate(results, 1):
            model_name = result.get('model', 'unknown')
            prompt_id = result.get('prompt_id', 'unknown')
            
            # Get prompt text - might be in different fields
            prompt_text = result.get('prompt_text')
            if not prompt_text:
                # Try to get from PROMPTS if it's a standard prompt
                prompt_mapping = {
                    'reasoning': 'Explain recursion in programming with a simple example.',
                    'code': 'Write a Python function to check if a string is a palindrome.',
                    'summary': 'Summarize the key principles of object-oriented programming in 2-3 sentences.'
                }
                prompt_text = prompt_mapping.get(prompt_id, prompt_id)
            
            response_text = result.get('response', '')
            
            print(f"[{i}/{len(results)}] Evaluating {model_name} / {prompt_id}...")
            
            scores = evaluate_response(model, prompt_text, response_text)
            
            if scores:
                evaluation = {
                    'backend': result.get('backend', backend.lower()),
                    'model': model_name,
                    'prompt_id': prompt_id,
                    'timestamp': result.get('timestamp'),
                    'judge': 'gemini-2.5-flash',
                    **scores
                }
                all_evaluations.append(evaluation)
                print(f"    ✓ Average score: {scores['average']:.1f}/10")
            else:
                print(f"    ✗ Evaluation failed")
    
    # Save combined evaluations
    output_file = f"gemini_evaluations_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_evaluations, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Saved {len(all_evaluations)} evaluations to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print summary by backend and model
    print("Summary by Backend and Model:")
    print("-" * 70)
    
    backends = {}
    for ev in all_evaluations:
        backend = ev['backend']
        model = ev['model']
        key = f"{backend}/{model}"
        
        if key not in backends:
            backends[key] = []
        backends[key].append(ev['average'])
    
    for key, scores in sorted(backends.items()):
        avg = sum(scores) / len(scores)
        backend, model = key.split('/', 1)
        print(f"\n[{backend.upper()}] {model}:")
        print(f"  Average Quality: {avg:.2f}/10 ({len(scores)} evaluations)")
    
    print()

if __name__ == '__main__':
    main()
