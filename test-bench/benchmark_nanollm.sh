#!/usr/bin/env bash
#
# NanoLLM Benchmark Script  
# Benchmarks LLMs using NanoLLM in jetson-containers (similar to MLC benchmark pattern)
#
# Usage:
#   HUGGINGFACE_TOKEN=hf_xxx ./benchmark_nanollm.sh                    # Run all default models
#   HUGGINGFACE_TOKEN=hf_xxx ./benchmark_nanollm.sh model-name         # Run specific model
#
# Or source token first:
#   source ~/.config/benchmark/hf_token.sh
#   ./benchmark_nanollm.sh
#
set -ex

# Check for HuggingFace token
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "ERROR: HUGGINGFACE_TOKEN not set!"
    echo "Run: ./setup_hf_token.sh"
    echo "Or: source ~/.config/benchmark/hf_token.sh"
    exit 1
fi

: "${QUANTIZATION:=q4f16_ft}"
: "${MAX_NEW_TOKENS:=128}"
: "${OUTPUT_CSV:=/data/benchmarks/nano_llm.csv}"
: "${API:=mlc}"

# Test prompts (matching Python benchmark)
PROMPTS=(
    "Explain the concept of recursion in programming and provide a simple example. Why is it useful?"
    "Write a Python function to check if a string is a palindrome. Include error handling and documentation."
    "Summarize the key principles of object-oriented programming in 3-4 sentences."
    "If a train travels 120 km in 2 hours, then stops for 30 minutes, and continues for another 90 km at the same speed, what is the total journey time? Show your reasoning."
    "Create a function that finds the two numbers in an array that sum to a target value. Optimize for time complexity."
    "Explain the difference between machine learning and deep learning in simple terms suitable for a non-technical audience."
    "What are the ethical considerations when deploying AI models in healthcare? List and explain three key concerns."
)

function benchmark_model() {
    local model=$1
    local model_name=$(basename $model)
    
    echo ""
    echo "========================================"
    echo "Benchmarking: $model_name"
    echo "========================================"
    
    # Test each prompt
    for idx in "${!PROMPTS[@]}"; do
        local prompt="${PROMPTS[$idx]}"
        echo ""
        echo "Prompt $((idx+1))/${#PROMPTS[@]}: ${prompt:0:60}..."
        
        # Run using nano_llm's chat module with HF token
        jetson-containers run \
            -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
            dustynv/nano_llm:r36.4.0 \
            python3 -m nano_llm.chat \
                --api ${API} \
                --model ${model} \
                --quantization ${QUANTIZATION} \
                --max-new-tokens ${MAX_NEW_TOKENS} \
                --prompt "${prompt}"
    done
    
    echo "✓ Completed: $model_name"
}

# Main execution
if [ "$#" -gt 0 ]; then
    # Benchmark specific model
    benchmark_model "$1"
else
    # Benchmark default models (same as Ollama benchmark)
    benchmark_model "meta-llama/Llama-3.2-1B-Instruct"
    benchmark_model "meta-llama/Llama-3.2-3B-Instruct"
    benchmark_model "meta-llama/Llama-3.1-8B-Instruct"
    
    echo ""
    echo "========================================"
    echo "All benchmarks complete!"
    echo "========================================"
fi
