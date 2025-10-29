#!/usr/bin/env python3
"""
Comprehensive LLM Benchmarking Script for NanoLLM and Ollama
Benchmarks multiple models with detailed metrics and quality evaluation
"""

import json
import time
import psutil
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import re
import os

# Try to import OpenAI for judge evaluation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available for judge evaluation")

# Try to import Google Generative AI for Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not available")


class TegraStat:
    """Helper class to parse tegrastats for GPU/CPU usage"""
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get current Tegra stats"""
        try:
            result = subprocess.run(
                ['tegrastats', '--interval', '100'],
                capture_output=True,
                text=True,
                timeout=0.5
            )
            output = result.stdout
            
            stats = {
                'gpu_usage': 0,
                'cpu_usage': 0,
                'memory_usage': 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Parse GPU usage
            gpu_match = re.search(r'GR3D_FREQ (\d+)%', output)
            if gpu_match:
                stats['gpu_usage'] = int(gpu_match.group(1))
            
            # Parse CPU usage (average across cores)
            cpu_matches = re.findall(r'CPU \[(\d+)%', output)
            if cpu_matches:
                stats['cpu_usage'] = sum(int(x) for x in cpu_matches) / len(cpu_matches)
            
            # Parse memory
            mem_match = re.search(r'RAM (\d+)/(\d+)MB', output)
            if mem_match:
                used = int(mem_match.group(1))
                total = int(mem_match.group(2))
                stats['memory_usage'] = (used / total) * 100
            
            return stats
        except Exception as e:
            # Fallback to psutil
            return {
                'gpu_usage': 0,  # Not available via psutil
                'cpu_usage': psutil.cpu_percent(interval=0.1),
                'memory_usage': psutil.virtual_memory().percent,
                'timestamp': datetime.now().isoformat()
            }


class LLMBenchmark:
    """Main benchmarking class"""
    
    # Test prompts covering different capabilities
    PROMPTS = [
        {
            "id": "reasoning_1",
            "category": "reasoning",
            "prompt": "Explain the concept of recursion in programming and provide a simple example. Why is it useful?"
        },
        {
            "id": "code_1",
            "category": "code",
            "prompt": "Write a Python function to check if a string is a palindrome. Include error handling and documentation."
        },
        {
            "id": "summarization_1",
            "category": "summarization",
            "prompt": "Summarize the key principles of object-oriented programming in 3-4 sentences."
        },
        {
            "id": "reasoning_2",
            "category": "reasoning",
            "prompt": "If a train travels 120 km in 2 hours, then stops for 30 minutes, and continues for another 90 km at the same speed, what is the total journey time? Show your reasoning."
        },
        {
            "id": "code_2",
            "category": "code",
            "prompt": "Create a function that finds the two numbers in an array that sum to a target value. Optimize for time complexity."
        },
        {
            "id": "summarization_2",
            "category": "summarization",
            "prompt": "Explain the difference between machine learning and deep learning in simple terms suitable for a non-technical audience."
        },
        {
            "id": "reasoning_3",
            "category": "reasoning",
            "prompt": "What are the ethical considerations when deploying AI models in healthcare? List and explain three key concerns."
        }
    ]
    
    MODELS = [
        "llama3.2:3b",     # Llama 3.2 3B
        "llama3.1:8b",     # Llama 3.1 8B
        "gemma3:4b",       # Gemma 3 4B
    ]
    
    def __init__(self):
        self.results = []
        self.quality_scores = []
        self.nanollm_url = "http://localhost:8080"
        self.ollama_url = "http://localhost:11434"
        self.available_judge_models = []
        self.use_nanollm_containers = False  # Disabled - models need pre-compilation
        self.nanollm_container = "dustynv/nano_llm:r36.4.0"
        
        # Configure Gemini if API key is available
        self.gemini_model = None
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if GEMINI_AVAILABLE and gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                print("✓ Gemini API configured (gemini-2.5-flash)")
            except Exception as e:
                print(f"Warning: Failed to configure Gemini: {e}")
        
        # Create output directories
        Path("plots").mkdir(exist_ok=True)
        
    def unload_ollama_model(self, model: str):
        """Unload a model from Ollama to free memory"""
        try:
            # Send a request with keep_alive=0 to unload the model
            requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": 0
                },
                timeout=10
            )
            print(f"    ✓ Unloaded model {model}")
        except Exception as e:
            print(f"    Warning: Failed to unload {model}: {e}")
    
    def get_available_judge_models(self) -> List[str]:
        """Get list of available Ollama models for judging"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                # Prefer smaller, faster models for judging
                judge_preferences = [
                    "llama3.2:3b", "llama3.1:8b-instruct-q2_K", 
                    "llama3.2:1b", "llama3.1:8b"
                ]
                available = [m for m in judge_preferences if m in model_names]
                if not available and model_names:
                    # Use first available model as fallback
                    available = [model_names[0]]
                return available
        except:
            pass
        return []
        
    def check_servers(self) -> Dict[str, bool]:
        """Check if NanoLLM and Ollama servers are running"""
        status = {}
        
        # Check NanoLLM - either via container or server
        if self.use_nanollm_containers:
            # Check if jetson-containers is available
            try:
                result = subprocess.run(['which', 'jetson-containers'], 
                                      capture_output=True, text=True)
                status['nanollm'] = (result.returncode == 0)
            except:
                status['nanollm'] = False
        else:
            # Check NanoLLM server
            try:
                response = requests.get(f"{self.nanollm_url}/health", timeout=5)
                status['nanollm'] = response.status_code == 200
            except:
                status['nanollm'] = False
            
        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            status['ollama'] = response.status_code == 200
        except:
            status['ollama'] = False
            
        return status
    
    def benchmark_nanollm(self, model: str, prompt: str, prompt_id: str, category: str) -> Dict[str, Any]:
        """Benchmark a single prompt on NanoLLM using jetson-containers"""
        print(f"  [NanoLLM] Testing {model} with prompt {prompt_id}...")
        
        # Check for HF token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("    Error: HUGGINGFACE_TOKEN not set. Run ./setup_hf_token.sh")
            return None
        
        model_name = model.split('/')[-1]
        
        # Get system stats before
        stats_before = TegraStat.get_stats()
        
        # Create temporary Python script to capture metrics
        temp_script = f'''
import json
import time
import sys
from datetime import datetime

# Redirect stderr to avoid cluttering output
import logging
logging.basicConfig(level=logging.ERROR)

try:
    from nano_llm import NanoLLM, ChatHistory
    
    model = "{model}"
    prompt = """{prompt}"""
    
    print("Loading model...", file=sys.stderr)
    load_start = time.time()
    
    llm = NanoLLM.from_pretrained(
        model,
        api='mlc',
        quantization='q4f16_ft'
    )
    
    load_time = time.time() - load_start
    print(f"Model loaded in {{load_time:.2f}}s", file=sys.stderr)
    
    chat_history = ChatHistory(llm)
    
    start_time = time.time()
    first_token_time = None
    tokens_generated = 0
    response_text = ""
    
    for token in chat_history.append(role='user', msg=prompt, use_cache=False):
        if first_token_time is None:
            first_token_time = time.time() - start_time
        tokens_generated += 1
        response_text += token
    
    total_time = time.time() - start_time
    
    result = {{
        "load_time": load_time,
        "total_time": total_time,
        "first_token_time": first_token_time or 0,
        "tokens": tokens_generated,
        "response": response_text
    }}
    
    print(json.dumps(result))
    
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
'''
        
        # Write temp script
        temp_file = '/tmp/nanollm_bench.py'
        with open(temp_file, 'w') as f:
            f.write(temp_script)
        
        try:
            # Run via jetson-containers with HF token
            start_time = time.time()
            result = subprocess.run(
                [
                    'jetson-containers', 'run', '--rm',
                    '-v', '/tmp:/tmp',
                    '-e', f'HUGGINGFACE_TOKEN={hf_token}',
                    self.nanollm_container,
                    'python3', temp_file
                ],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes max
            )
            
            if result.returncode != 0:
                print(f"    Error: Container exited with code {result.returncode}")
                print(f"    Stderr: {result.stderr[-500:]}")  # Last 500 chars
                return None
            
            # Parse output (last line should be JSON)
            output_lines = [line for line in result.stdout.strip().split('\n') if line]
            if not output_lines:
                print(f"    Error: No output from container")
                return None
                
            try:
                bench_result = json.loads(output_lines[-1])
            except json.JSONDecodeError:
                print(f"    Error: Could not parse JSON from output")
                print(f"    Last line: {output_lines[-1][:200]}")
                return None
            
            if 'error' in bench_result:
                print(f"    Error: {bench_result['error']}")
                return None
            
        except subprocess.TimeoutExpired:
            print(f"    Error: Benchmark timed out after 10 minutes")
            return None
        except Exception as e:
            print(f"    Error: {e}")
            return None
        
        # Get system stats after
        stats_after = TegraStat.get_stats()
        
        # Calculate metrics
        throughput = bench_result['tokens'] / bench_result['total_time'] if bench_result['total_time'] > 0 else 0
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'backend': 'nanollm',
            'model': model,
            'prompt_id': prompt_id,
            'prompt_category': category,
            'prompt': prompt,
            'response': bench_result['response'],
            'total_latency_s': bench_result['total_time'],
            'first_token_latency_s': bench_result['first_token_time'],
            'tokens_generated': bench_result['tokens'],
            'tokens_per_sec': throughput,
            'model_load_time_s': bench_result['load_time'],
            'gpu_usage_start': stats_before['gpu_usage'],
            'gpu_usage_end': stats_after['gpu_usage'],
            'cpu_usage_start': stats_before['cpu_usage'],
            'cpu_usage_end': stats_after['cpu_usage'],
            'memory_usage_start': stats_before['memory_usage'],
            'memory_usage_end': stats_after['memory_usage']
        }
        
        print(f"    ✓ Completed in {bench_result['total_time']:.2f}s, {throughput:.2f} tokens/s")
        return result
    
    def benchmark_nanollm_old(self, model: str, prompt: str, prompt_id: str, category: str) -> Dict[str, Any]:
        """Benchmark a single prompt on NanoLLM (old REST API method)"""
        print(f"  [NanoLLM] Testing {model} with prompt {prompt_id}...")
        
        # Get model name for NanoLLM (strip org prefix)
        model_name = model.split('/')[-1]
        
        # Measure model load time
        load_start = time.time()
        try:
            # Check if model is loaded
            response = requests.get(f"{self.nanollm_url}/models", timeout=10)
            loaded_models = response.json() if response.status_code == 200 else []
        except:
            loaded_models = []
        
        model_load_time = time.time() - load_start
        
        # Get system stats before
        stats_before = TegraStat.get_stats()
        
        # Run inference
        start_time = time.time()
        first_token_time = None
        tokens_generated = 0
        response_text = ""
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 512,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.nanollm_url}/v1/completions",
                json=payload,
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'text' in data:
                            response_text += data['text']
                            tokens_generated += 1
                    except:
                        continue
            
            total_time = time.time() - start_time
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
        
        # Get system stats after
        stats_after = TegraStat.get_stats()
        
        # Calculate metrics
        throughput = tokens_generated / total_time if total_time > 0 else 0
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'backend': 'nanollm',
            'model': model,
            'prompt_id': prompt_id,
            'prompt_category': category,
            'prompt': prompt,
            'response': response_text,
            'total_latency_s': total_time,
            'first_token_latency_s': first_token_time or 0,
            'tokens_generated': tokens_generated,
            'tokens_per_sec': throughput,
            'model_load_time_s': model_load_time,
            'gpu_usage_start': stats_before['gpu_usage'],
            'gpu_usage_end': stats_after['gpu_usage'],
            'cpu_usage_start': stats_before['cpu_usage'],
            'cpu_usage_end': stats_after['cpu_usage'],
            'memory_usage_start': stats_before['memory_usage'],
            'memory_usage_end': stats_after['memory_usage']
        }
        
        print(f"    ✓ Completed in {total_time:.2f}s, {throughput:.2f} tokens/s")
        return result
    
    def benchmark_ollama(self, model: str, prompt: str, prompt_id: str, category: str) -> Dict[str, Any]:
        """Benchmark a single prompt on Ollama"""
        print(f"  [Ollama] Testing {model} with prompt {prompt_id}...")
        
        # Get model name for Ollama (convert format)
        model_name = model.split('/')[-1].lower().replace('-instruct', ':instruct')
        
        # Measure model load time
        load_start = time.time()
        try:
            # Pull model if not available
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                timeout=300
            )
        except:
            pass
        
        model_load_time = time.time() - load_start
        
        # Get system stats before
        stats_before = TegraStat.get_stats()
        
        # Run inference
        start_time = time.time()
        first_token_time = None
        tokens_generated = 0
        response_text = ""
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 512,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            response_text += data['response']
                            tokens_generated += 1
                        
                        if data.get('done', False):
                            break
                    except:
                        continue
            
            total_time = time.time() - start_time
            
        except Exception as e:
            print(f"    Error: {e}")
            return None
        
        # Get system stats after
        stats_after = TegraStat.get_stats()
        
        # Calculate metrics
        throughput = tokens_generated / total_time if total_time > 0 else 0
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'backend': 'ollama',
            'model': model,
            'prompt_id': prompt_id,
            'prompt_category': category,
            'prompt': prompt,
            'response': response_text,
            'total_latency_s': total_time,
            'first_token_latency_s': first_token_time or 0,
            'tokens_generated': tokens_generated,
            'tokens_per_sec': throughput,
            'model_load_time_s': model_load_time,
            'gpu_usage_start': stats_before['gpu_usage'],
            'gpu_usage_end': stats_after['gpu_usage'],
            'cpu_usage_start': stats_before['cpu_usage'],
            'cpu_usage_end': stats_after['cpu_usage'],
            'memory_usage_start': stats_before['memory_usage'],
            'memory_usage_end': stats_after['memory_usage']
        }
        
        print(f"    ✓ Completed in {total_time:.2f}s, {throughput:.2f} tokens/s")
        return result
    
    def evaluate_response_quality(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM-as-Judge to evaluate response quality"""
        print(f"  Evaluating quality for {result['backend']}/{result['model']}/{result['prompt_id']}...")
        
        judge_prompt = f"""You are an expert AI evaluator. Evaluate the following LLM response on a scale of 1-10 for each criterion.

Prompt: {result['prompt']}

Response: {result['response']}

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
            # Priority 1: Try Gemini API (fast and good quality)
            if self.gemini_model:
                print("    Using Gemini judge...")
                try:
                    response = self.gemini_model.generate_content(judge_prompt)
                    scores_text = response.text.strip()
                    # Extract JSON from response
                    json_start = scores_text.find('{')
                    json_end = scores_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        scores = json.loads(scores_text[json_start:json_end])
                    else:
                        raise ValueError("No JSON found in Gemini response")
                except Exception as e:
                    print(f"    Gemini failed: {e}, falling back...")
                    scores = None
                
                if scores:
                    quality_result = {
                        'timestamp': datetime.now().isoformat(),
                        'backend': result['backend'],
                        'model': result['model'],
                        'prompt_id': result['prompt_id'],
                        'prompt_category': result['prompt_category'],
                        'judge': 'gemini-2.5-flash',
                        'correctness': scores.get('correctness', 5),
                        'relevance': scores.get('relevance', 5),
                        'completeness': scores.get('completeness', 5),
                        'coherence': scores.get('coherence', 5),
                        'reasoning_quality': scores.get('reasoning_quality', 5),
                        'average_quality': np.mean([
                            scores.get('correctness', 5),
                            scores.get('relevance', 5),
                            scores.get('completeness', 5),
                            scores.get('coherence', 5),
                            scores.get('reasoning_quality', 5)
                        ])
                    }
                    print(f"    Quality scores: {quality_result['average_quality']:.1f}/10 avg")
                    return quality_result
            
            # Priority 2: Try OpenAI API (GPT-4o-mini)
            if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
                print("    Using OpenAI judge...")
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert AI response evaluator. Always respond with valid JSON."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                
                scores_text = response.choices[0].message.content.strip()
                # Extract JSON from response
                json_start = scores_text.find('{')
                json_end = scores_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    scores = json.loads(scores_text[json_start:json_end])
                else:
                    raise ValueError("No JSON found in response")
                    
            else:
                # Priority 3: Fallback to local Ollama judge
                print("    Using local Ollama judge...")
                
                # Get available judge models if not already cached
                if not self.available_judge_models:
                    self.available_judge_models = self.get_available_judge_models()
                
                if not self.available_judge_models:
                    raise ValueError("No Ollama models available for judging")
                
                # Try available models
                response = None
                used_model = None
                
                for judge_model in self.available_judge_models:
                    try:
                        response = requests.post(
                            f"{self.ollama_url}/api/generate",
                            json={
                                "model": judge_model,
                                "prompt": judge_prompt,
                                "stream": False,
                                "options": {"temperature": 0.3}
                            },
                            timeout=60
                        )
                        if response.status_code == 200:
                            used_model = judge_model
                            break
                    except:
                        continue
                
                if response is None or response.status_code != 200:
                    raise ValueError(f"No available Ollama judge model found. Tried: {self.available_judge_models}")
                
                if response.status_code == 200:
                    response_text = response.json().get('response', '')
                    # Extract JSON from response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        scores = json.loads(response_text[json_start:json_end])
                    else:
                        raise ValueError("No JSON found in response")
                else:
                    raise ValueError(f"Ollama request failed: {response.status_code}")
            
            quality_result = {
                'timestamp': datetime.now().isoformat(),
                'backend': result['backend'],
                'model': result['model'],
                'prompt_id': result['prompt_id'],
                'prompt_category': result['prompt_category'],
                'correctness': scores.get('correctness', 5),
                'relevance': scores.get('relevance', 5),
                'completeness': scores.get('completeness', 5),
                'coherence': scores.get('coherence', 5),
                'reasoning_quality': scores.get('reasoning_quality', 5),
                'average_quality': np.mean([
                    scores.get('correctness', 5),
                    scores.get('relevance', 5),
                    scores.get('completeness', 5),
                    scores.get('coherence', 5),
                    scores.get('reasoning_quality', 5)
                ])
            }
            
            print(f"    Quality scores: {quality_result['average_quality']:.1f}/10 avg")
            return quality_result
            
        except Exception as e:
            print(f"    Warning: Quality evaluation failed: {e}")
            # Return default scores
            return {
                'timestamp': datetime.now().isoformat(),
                'backend': result['backend'],
                'model': result['model'],
                'prompt_id': result['prompt_id'],
                'prompt_category': result['prompt_category'],
                'correctness': 5,
                'relevance': 5,
                'completeness': 5,
                'coherence': 5,
                'reasoning_quality': 5,
                'average_quality': 5.0
            }
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*60)
        print("Starting LLM Benchmark Suite")
        print("="*60)
        
        # Check server status
        status = self.check_servers()
        print(f"\nServer Status:")
        nanollm_status = '✓ jetson-containers available' if self.use_nanollm_containers else ('✓ Running' if status['nanollm'] else '✗ Not available')
        print(f"  NanoLLM: {nanollm_status}")
        print(f"  Ollama:  {'✓ Running' if status['ollama'] else '✗ Not available'}")
        
        # Check for judge models
        if status['ollama']:
            self.available_judge_models = self.get_available_judge_models()
            if self.available_judge_models:
                print(f"  Judge Model: {self.available_judge_models[0]}")
            else:
                print(f"  Judge Model: ⚠ None available (will use default scores)")
        elif OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            print(f"  Judge Model: GPT-4o-mini (via OpenAI API)")
        
        if not any(status.values()):
            print("\nError: No servers are running. Please start NanoLLM or Ollama.")
            return
        
        print(f"\nModels to benchmark: {len(self.MODELS)}")
        print(f"Prompts per model: {len(self.PROMPTS)}")
        print(f"Backends: {['NanoLLM', 'Ollama']}")
        
        total_runs = len(self.MODELS) * len(self.PROMPTS) * sum(status.values())
        print(f"Total benchmark runs: {total_runs}\n")
        
        run_count = 0
        results_to_evaluate = []  # Defer quality evaluation until the end
        
        for model in self.MODELS:
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            print(f"{'='*60}")
            
            for prompt_data in self.PROMPTS:
                prompt_id = prompt_data['id']
                category = prompt_data['category']
                prompt = prompt_data['prompt']
                
                print(f"\nPrompt [{prompt_id}] ({category}):")
                print(f"  {prompt[:80]}...")
                
                # Test on NanoLLM
                if status['nanollm']:
                    result = self.benchmark_nanollm(model, prompt, prompt_id, category)
                    if result:
                        self.results.append(result)
                        results_to_evaluate.append(result)
                        run_count += 1
                        print(f"  Progress: {run_count}/{total_runs}")
                
                # Test on Ollama
                if status['ollama']:
                    result = self.benchmark_ollama(model, prompt, prompt_id, category)
                    if result:
                        self.results.append(result)
                        results_to_evaluate.append(result)
                        run_count += 1
                        print(f"  Progress: {run_count}/{total_runs}")
            
            # Unload model after testing to free memory
            if status['ollama']:
                print(f"\n  Unloading {model} from memory...")
                self.unload_ollama_model(model)
        
        print(f"\n{'='*60}")
        print(f"Benchmarking complete! {len(self.results)} runs completed.")
        print(f"{'='*60}\n")
        
        # Now evaluate all responses with the judge model
        if results_to_evaluate:
            print(f"\n{'='*60}")
            print(f"Evaluating response quality for {len(results_to_evaluate)} results...")
            print(f"{'='*60}\n")
            
            for i, result in enumerate(results_to_evaluate, 1):
                print(f"[{i}/{len(results_to_evaluate)}] Evaluating {result['backend']}/{result['model']}/{result['prompt_id']}...")
                quality = self.evaluate_response_quality(result)
                self.quality_scores.append(quality)
            
            print(f"\n✓ Quality evaluation complete!\n")
    
    def save_results(self):
        """Save results to JSON files"""
        print("Saving results...")
        
        # Save benchmark results
        with open('benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("  ✓ Saved benchmark_results.json")
        
        # Save quality scores
        with open('quality_scores.json', 'w') as f:
            json.dump(self.quality_scores, f, indent=2)
        print("  ✓ Saved quality_scores.json")
    
    def analyze_and_visualize(self):
        """Analyze results and create visualizations"""
        print("\nAnalyzing results and creating visualizations...")
        
        if not self.results:
            print("No results to analyze!")
            return
        
        # Convert to DataFrames
        df_perf = pd.DataFrame(self.results)
        df_quality = pd.DataFrame(self.quality_scores)
        
        # Merge performance and quality data
        df_merged = df_perf.merge(
            df_quality,
            on=['backend', 'model', 'prompt_id', 'prompt_category'],
            how='left',
            suffixes=('', '_quality')
        )
        
        # Calculate summary statistics
        summary = df_merged.groupby(['backend', 'model']).agg({
            'total_latency_s': ['mean', 'std'],
            'first_token_latency_s': ['mean', 'std'],
            'tokens_per_sec': ['mean', 'std'],
            'tokens_generated': ['mean', 'std'],
            'gpu_usage_end': ['mean', 'std'],
            'cpu_usage_end': ['mean', 'std'],
            'correctness': ['mean', 'std'],
            'relevance': ['mean', 'std'],
            'completeness': ['mean', 'std'],
            'coherence': ['mean', 'std'],
            'reasoning_quality': ['mean', 'std'],
            'average_quality': ['mean', 'std']
        }).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Save summary to CSV
        summary.to_csv('results_summary.csv', index=False)
        print("  ✓ Saved results_summary.csv")
        
        # Create visualizations
        self._create_performance_plots(df_merged)
        self._create_quality_plots(df_merged)
        
        # Print summary
        self._print_summary(summary)
    
    def _create_performance_plots(self, df: pd.DataFrame):
        """Create performance comparison plots"""
        
        # 1. Latency vs Throughput by Backend and Model
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group data
        grouped = df.groupby(['backend', 'model']).agg({
            'total_latency_s': 'mean',
            'tokens_per_sec': 'mean'
        }).reset_index()
        
        # Latency comparison
        backends = grouped['backend'].unique()
        x = np.arange(len(grouped['model'].unique()))
        width = 0.35
        
        for i, backend in enumerate(backends):
            data = grouped[grouped['backend'] == backend]
            offset = width * (i - 0.5)
            ax1.bar(x + offset, data['total_latency_s'], width, label=backend)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Mean Latency (seconds)')
        ax1.set_title('Average Total Latency by Model and Backend')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.split('/')[-1] for m in grouped['model'].unique()], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Throughput comparison
        for i, backend in enumerate(backends):
            data = grouped[grouped['backend'] == backend]
            offset = width * (i - 0.5)
            ax2.bar(x + offset, data['tokens_per_sec'], width, label=backend)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Average Throughput by Model and Backend')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.split('/')[-1] for m in grouped['model'].unique()], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved plots/performance_comparison.png")
        
        # 2. First Token Latency
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped_ftl = df.groupby(['backend', 'model'])['first_token_latency_s'].mean().reset_index()
        
        for i, backend in enumerate(backends):
            data = grouped_ftl[grouped_ftl['backend'] == backend]
            offset = width * (i - 0.5)
            ax.bar(x + offset, data['first_token_latency_s'], width, label=backend)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('First Token Latency (seconds)')
        ax.set_title('Average First Token Latency by Model and Backend')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in grouped_ftl['model'].unique()], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/first_token_latency.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved plots/first_token_latency.png")
    
    def _create_quality_plots(self, df: pd.DataFrame):
        """Create quality comparison plots"""
        
        # Radar plot for quality metrics
        quality_cols = ['correctness', 'relevance', 'completeness', 'coherence', 'reasoning_quality']
        
        # Get mean scores by backend and model
        quality_data = df.groupby(['backend', 'model'])[quality_cols].mean()
        
        # Create a radar plot for each backend
        num_vars = len(quality_cols)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
        
        backends = df['backend'].unique()
        for idx, backend in enumerate(backends):
            ax = axes[idx]
            
            for model in df['model'].unique():
                if (backend, model) in quality_data.index:
                    values = quality_data.loc[(backend, model)].tolist()
                    values += values[:1]  # Complete the circle
                    
                    model_label = model.split('/')[-1]
                    ax.plot(angles, values, 'o-', linewidth=2, label=model_label)
                    ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([col.replace('_', ' ').title() for col in quality_cols])
            ax.set_ylim(0, 10)
            ax.set_title(f'{backend.upper()} Quality Scores', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/quality_radar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved plots/quality_radar.png")
        
        # Bar chart for average quality
        fig, ax = plt.subplots(figsize=(10, 6))
        
        grouped = df.groupby(['backend', 'model'])['average_quality'].mean().reset_index()
        
        backends = grouped['backend'].unique()
        x = np.arange(len(grouped['model'].unique()))
        width = 0.35
        
        for i, backend in enumerate(backends):
            data = grouped[grouped['backend'] == backend]
            offset = width * (i - 0.5)
            ax.bar(x + offset, data['average_quality'], width, label=backend)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Quality Score (1-10)')
        ax.set_title('Average Quality Score by Model and Backend')
        ax.set_xticks(x)
        ax.set_xticklabels([m.split('/')[-1] for m in grouped['model'].unique()], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig('plots/average_quality.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved plots/average_quality.png")
    
    def _print_summary(self, summary: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for _, row in summary.iterrows():
            backend = row['backend']
            model = row['model'].split('/')[-1]
            
            print(f"\n{backend.upper()} - {model}")
            print("-" * 60)
            print(f"  Performance Metrics:")
            print(f"    Latency:          {row['total_latency_s_mean']:.2f}s ± {row['total_latency_s_std']:.2f}s")
            print(f"    First Token:      {row['first_token_latency_s_mean']:.3f}s ± {row['first_token_latency_s_std']:.3f}s")
            print(f"    Throughput:       {row['tokens_per_sec_mean']:.2f} ± {row['tokens_per_sec_std']:.2f} tokens/s")
            print(f"    Tokens Generated: {row['tokens_generated_mean']:.0f} ± {row['tokens_generated_std']:.0f}")
            print(f"  Resource Usage:")
            print(f"    GPU Usage:        {row['gpu_usage_end_mean']:.1f}% ± {row['gpu_usage_end_std']:.1f}%")
            print(f"    CPU Usage:        {row['cpu_usage_end_mean']:.1f}% ± {row['cpu_usage_end_std']:.1f}%")
            print(f"  Quality Scores (1-10):")
            print(f"    Correctness:      {row['correctness_mean']:.1f} ± {row['correctness_std']:.1f}")
            print(f"    Relevance:        {row['relevance_mean']:.1f} ± {row['relevance_std']:.1f}")
            print(f"    Completeness:     {row['completeness_mean']:.1f} ± {row['completeness_std']:.1f}")
            print(f"    Coherence:        {row['coherence_mean']:.1f} ± {row['coherence_std']:.1f}")
            print(f"    Reasoning:        {row['reasoning_quality_mean']:.1f} ± {row['reasoning_quality_std']:.1f}")
            print(f"    Average:          {row['average_quality_mean']:.1f} ± {row['average_quality_std']:.1f}")
        
        print("\n" + "="*80)
        print("Files generated:")
        print("  - benchmark_results.json  (raw performance data)")
        print("  - quality_scores.json     (LLM-as-Judge evaluations)")
        print("  - results_summary.csv     (aggregated statistics)")
        print("  - plots/performance_comparison.png")
        print("  - plots/first_token_latency.png")
        print("  - plots/quality_radar.png")
        print("  - plots/average_quality.png")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    benchmark = LLMBenchmark()
    
    try:
        # Run benchmarks
        benchmark.run_benchmarks()
        
        # Save results
        benchmark.save_results()
        
        # Analyze and visualize
        benchmark.analyze_and_visualize()
        
        print("\n✅ Benchmarking completed successfully!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        if benchmark.results:
            print("Saving partial results...")
            benchmark.save_results()
            benchmark.analyze_and_visualize()
    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        if benchmark.results:
            print("Saving partial results...")
            benchmark.save_results()


if __name__ == "__main__":
    main()
