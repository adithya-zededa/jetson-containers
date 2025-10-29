#!/usr/bin/env python3
import os
import sys
import json
import socket
import datetime
import argparse
import resource
import time
import requests
import tiktoken

# parse model arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="llama3.1:8b")
parser.add_argument("--prompt", action='append', nargs='*')
parser.add_argument("--chat", action="store_true")
parser.add_argument("--streaming", action="store_true", default=True)
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--max-num-prompts", type=int, default=None)
parser.add_argument('--save', type=str, default='', help='CSV file to save benchmarking results to')
parser.add_argument('--base-url', type=str, default='http://localhost:11434', help='Ollama API base URL')

args = parser.parse_args()

# assign default prompts
if not args.prompt:
    if args.chat:  # https://modal.com/docs/guide/ex/vllm_inference
        args.prompt = [
            "What is the meaning of life?",
            "How many points did you list out?",
            "What is the weather forecast today?",
            "What is the fable involving a fox and grapes?",
            "What's a good recipe for making tabouli?",
            "What is the product of 9 and 8?",
            "If a train travels 120 miles in 2 hours, what is its average speed?",
        ]
    else:
        args.prompt = [
            "Once upon a time,",
            "A great place to live is",
            "In a world where dreams are shared,",
            "The weather forecast today is",
            "Large language models are",
            "Space exploration is exciting",
            "The history of the Hoover Dam is",
            "San Fransisco is a city in",
            "To train for running a marathon,",
            "A recipe for making tabouli is"
        ]
else:
    args.prompt = [x[0] for x in args.prompt]
    
print(args)

def load_prompts(prompts):
    """
    Load prompts from a list of txt or json files
    (or if these are strings, just return the strings)
    """
    prompt_list = []
    
    for prompt in prompts:
        ext = os.path.splitext(prompt)[1]
        
        if ext == '.json':
            with open(prompt) as file:
                json_prompts = json.load(file)
            for json_prompt in json_prompts:
                if isinstance(json_prompt, dict):
                    prompt_list.append(json_prompt)  # json_prompt['text']
                elif isinstance(json_prompt, str):
                    prompt_list.append(json_prompt)
                else:
                    raise TypeError(f"{type(json_prompt)}")
        elif ext == '.txt':
            with open(prompt) as file:
                prompt_list.append(file.read())
        else:
            prompt_list.append(prompt)
            
    return prompt_list
    
# load prompts if given a txt/json file
args.prompt = load_prompts(args.prompt)

if args.max_num_prompts:
    args.prompt = args.prompt[:args.max_num_prompts]

# Initialize tokenizer for token counting (use cl100k_base as approximation)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    print(f"Warning: Could not load tiktoken encoder ({e}), will use API token counts")
    tokenizer = None

def count_tokens(text):
    """Count tokens in text using tiktoken or character approximation"""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Rough approximation: ~4 chars per token
        return len(text) // 4

def generate(prompt, stats):
    """Generate text using Ollama API with detailed metrics"""
    
    url = f"{args.base_url}/api/generate"
    
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": args.streaming,
        "options": {
            "num_predict": args.max_new_tokens,
        }
    }
    
    response_text = ""
    total_duration = 0
    load_duration = 0
    prompt_eval_count = 0
    prompt_eval_duration = 0
    eval_count = 0
    eval_duration = 0
    
    try:
        response = requests.post(url, json=payload, stream=args.streaming)
        response.raise_for_status()
        
        if args.streaming:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        response_text += chunk['response']
                        print(chunk['response'], end='', flush=True)
                    
                    if chunk.get('done', False):
                        # Extract metrics from final chunk
                        total_duration = chunk.get('total_duration', 0)
                        load_duration = chunk.get('load_duration', 0)
                        prompt_eval_count = chunk.get('prompt_eval_count', 0)
                        prompt_eval_duration = chunk.get('prompt_eval_duration', 0)
                        eval_count = chunk.get('eval_count', 0)
                        eval_duration = chunk.get('eval_duration', 0)
        else:
            result = response.json()
            response_text = result.get('response', '')
            print(response_text)
            
            total_duration = result.get('total_duration', 0)
            load_duration = result.get('load_duration', 0)
            prompt_eval_count = result.get('prompt_eval_count', 0)
            prompt_eval_duration = result.get('prompt_eval_duration', 0)
            eval_count = result.get('eval_count', 0)
            eval_duration = result.get('eval_duration', 0)
    
    except Exception as e:
        print(f"Error during generation: {e}")
        raise
    
    # Convert nanoseconds to seconds
    stats['total_time'] = total_duration / 1e9 if total_duration > 0 else 0
    stats['load_time'] = load_duration / 1e9 if load_duration > 0 else 0
    stats['prefill_time'] = prompt_eval_duration / 1e9 if prompt_eval_duration > 0 else 0.001
    stats['decode_time'] = eval_duration / 1e9 if eval_duration > 0 else 0.001
    
    # Token counts
    stats['input_tokens'] = prompt_eval_count if prompt_eval_count > 0 else count_tokens(prompt)
    stats['output_tokens'] = eval_count if eval_count > 0 else count_tokens(response_text)
    
    # Calculate rates
    stats['prefill_rate'] = stats['input_tokens'] / stats['prefill_time'] if stats['prefill_time'] > 0 else 0
    stats['decode_rate'] = stats['output_tokens'] / stats['decode_time'] if stats['decode_time'] > 0 else 0
    
    return response_text

# Test Ollama connection and check/download model
print(f"-- Testing connection to Ollama at {args.base_url}")
try:
    response = requests.get(f"{args.base_url}/api/tags")
    response.raise_for_status()
    available_models = [model['name'] for model in response.json().get('models', [])]
    print(f"-- Available models: {', '.join(available_models)}")
    
    if args.model not in available_models:
        print(f"\n⚠️  Model '{args.model}' is not installed.")
        
        # Check if running interactively
        if sys.stdin.isatty():
            response = input("Download it now? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                print(f"📥 Downloading {args.model}...")
                try:
                    # Use Ollama API to pull the model
                    pull_response = requests.post(
                        f"{args.base_url}/api/pull",
                        json={"name": args.model},
                        stream=True
                    )
                    pull_response.raise_for_status()
                    
                    # Stream the download progress
                    last_status = ""
                    for line in pull_response.iter_lines():
                        if line:
                            status = json.loads(line)
                            if 'status' in status:
                                status_msg = f"  {status['status']}"
                                if 'completed' in status and 'total' in status:
                                    pct = (status['completed'] / status['total']) * 100
                                    status_msg += f" ({pct:.1f}%)"
                                
                                # Only print on same line if status hasn't changed significantly
                                if status['status'] != last_status:
                                    if last_status:
                                        print()  # New line for status change
                                    print(status_msg, end='', flush=True)
                                    last_status = status['status']
                                else:
                                    # Update same line for progress
                                    print(f"\r{status_msg}", end='', flush=True)
                    
                    print()  # Final newline
                    print(f"✅ Successfully downloaded {args.model}")
                except Exception as pull_error:
                    print(f"❌ Failed to download {args.model}: {pull_error}")
                    sys.exit(1)
            else:
                print(f"⏭️  Skipping model {args.model}")
                sys.exit(0)
        else:
            # Non-interactive mode: try to pull automatically
            print(f"📥 Attempting to download {args.model} (non-interactive mode)...")
            try:
                pull_response = requests.post(
                    f"{args.base_url}/api/pull",
                    json={"name": args.model},
                    stream=True
                )
                pull_response.raise_for_status()
                
                last_status = ""
                for line in pull_response.iter_lines():
                    if line:
                        status = json.loads(line)
                        if 'status' in status:
                            status_msg = f"  {status['status']}"
                            if 'completed' in status and 'total' in status:
                                pct = (status['completed'] / status['total']) * 100
                                status_msg += f" ({pct:.1f}%)"
                            
                            # Only print on same line if status hasn't changed significantly
                            if status['status'] != last_status:
                                if last_status:
                                    print()  # New line for status change
                                print(status_msg, end='', flush=True)
                                last_status = status['status']
                            else:
                                # Update same line for progress
                                print(f"\r{status_msg}", end='', flush=True)
                
                print()  # Final newline
                print(f"✅ Successfully downloaded {args.model}")
            except Exception as pull_error:
                print(f"❌ Failed to download {args.model}: {pull_error}")
                sys.exit(1)
                
except Exception as e:
    print(f"Warning: Could not connect to Ollama API ({e})")
    print(f"Make sure Ollama is running at {args.base_url}")
    sys.exit(1)

print(f"\n-- Benchmarking {args.model}")

# benchmark inference
avg_stats = {}

for i, prompt in enumerate(args.prompt):
    stats = {}
    
    if isinstance(prompt, dict):
        stats['input_tokens'] = prompt.get('num_tokens', count_tokens(prompt['text']))
        prompt = prompt['text']
        
    print(f"\nPROMPT:  {prompt}\n")
    
    try:
        output = generate(prompt, stats)
    except Exception as e:
        print(f"\nError generating response: {e}")
        continue
                
    print(f"\n{args.model}:  input={stats['input_tokens']} output={stats['output_tokens']} prefill_time {stats['prefill_time']:.3f} sec, prefill_rate {stats['prefill_rate']:.1f} tokens/sec, decode_time {stats['decode_time']:.3f} sec, decode_rate {stats['decode_rate']:.1f} tokens/sec\n")

    # Skip first prompt in averaging (warmup)
    if i > 0:
        for key in stats:
            avg_stats[key] = avg_stats.get(key, 0) + stats[key] * (1.0 / (len(args.prompt) - 1))

avg_stats['input_tokens'] = int(round(avg_stats['input_tokens']))
avg_stats['output_tokens'] = int(round(avg_stats['output_tokens']))

print(f"AVERAGE OVER {len(args.prompt) - 1} RUNS  (input_tokens={avg_stats['input_tokens']}, output_tokens={avg_stats['output_tokens']})")
print(f"{args.model}:  prefill_time {avg_stats['prefill_time']:.3f} sec, prefill_rate {avg_stats['prefill_rate']:.1f} tokens/sec, decode_time {avg_stats['decode_time']:.3f} sec, decode_rate {avg_stats['decode_rate']:.1f} tokens/sec\n")

memory_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024  # https://stackoverflow.com/a/7669482

print(f"Peak memory usage:  {memory_usage:.2f} MB")

if args.save:
    if not os.path.isfile(args.save):  # csv header
        with open(args.save, 'w') as file:
            file.write(f"timestamp, hostname, api, model, precision, input_tokens, output_tokens, prefill_time, prefill_rate, decode_time, decode_rate, memory\n")
    with open(args.save, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')}, {socket.gethostname()}, ollama, ")
        
        # Extract precision/quantization from model name if available
        model_parts = args.model.split(':')
        precision = model_parts[1] if len(model_parts) > 1 else "unknown"
        
        file.write(f"{args.model}, {precision}, {avg_stats['input_tokens']}, {avg_stats['output_tokens']}, ")
        file.write(f"{avg_stats['prefill_time']}, {avg_stats['prefill_rate']}, {avg_stats['decode_time']}, {avg_stats['decode_rate']}, {memory_usage}\n")
    print(f"Saved results to:   {args.save}")
