#!/usr/bin/env bash
#
# Comprehensive LLM Benchmark Suite
# Runs Ollama and NanoLLM benchmarks, then evaluates with Gemini
#
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SKIP_OLLAMA=${SKIP_OLLAMA:-no}
SKIP_NANOLLM=${SKIP_NANOLLM:-no}
SKIP_EVALUATION=${SKIP_EVALUATION:-no}

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   Comprehensive LLM Benchmark Suite${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "This script will:"
echo "  1. Benchmark models on Ollama"
echo "  2. Benchmark models on NanoLLM (MLC)"
echo "  3. Evaluate all responses with Gemini"
echo "  4. Generate comparison reports"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check for Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    if [ -f ~/.config/benchmark/gemini_key.sh ]; then
        source ~/.config/benchmark/gemini_key.sh
        echo -e "${GREEN}✓ Loaded Gemini API key${NC}"
    else
        echo -e "${RED}✗ GEMINI_API_KEY not set${NC}"
        echo "  Run: source ~/.config/benchmark/gemini_key.sh"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Gemini API key found${NC}"
fi

# Check Ollama
if [ "$SKIP_OLLAMA" != "yes" ]; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama server running${NC}"
    else
        echo -e "${RED}✗ Ollama server not running${NC}"
        echo "  Start with: sudo systemctl start ollama"
        exit 1
    fi
fi

# Check jetson-containers
if [ "$SKIP_NANOLLM" != "yes" ]; then
    if which jetson-containers > /dev/null 2>&1; then
        echo -e "${GREEN}✓ jetson-containers available${NC}"
    else
        echo -e "${RED}✗ jetson-containers not found${NC}"
        exit 1
    fi
fi

echo ""

# ============================================================================
# Phase 1: Ollama Benchmark
# ============================================================================
if [ "$SKIP_OLLAMA" != "yes" ]; then
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}Phase 1: Ollama Benchmark${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    python3 quick_benchmark.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Ollama benchmark complete${NC}"
    else
        echo -e "${RED}✗ Ollama benchmark failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping Ollama benchmark (SKIP_OLLAMA=yes)${NC}"
    echo ""
fi

# ============================================================================
# Phase 2: NanoLLM Benchmark
# ============================================================================
if [ "$SKIP_NANOLLM" != "yes" ]; then
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}Phase 2: NanoLLM Benchmark${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    python3 benchmark_nanollm.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ NanoLLM benchmark complete${NC}"
    else
        echo -e "${YELLOW}⚠ NanoLLM benchmark failed (may need model compilation)${NC}"
        echo -e "${YELLOW}  Continuing with available results...${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping NanoLLM benchmark (SKIP_NANOLLM=yes)${NC}"
    echo ""
fi

# ============================================================================
# Phase 3: Gemini Evaluation
# ============================================================================
if [ "$SKIP_EVALUATION" != "yes" ]; then
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}Phase 3: Gemini Quality Evaluation${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
    
    python3 evaluate_with_gemini.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Quality evaluation complete${NC}"
    else
        echo -e "${RED}✗ Quality evaluation failed${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}Skipping evaluation (SKIP_EVALUATION=yes)${NC}"
    echo ""
fi

# ============================================================================
# Phase 4: Generate Summary Report
# ============================================================================
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}Phase 4: Generating Summary Report${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

python3 << 'PYTHON_SCRIPT'
import json
import glob
from pathlib import Path
from datetime import datetime

print("Analyzing benchmark results...\n")

# Find latest results
ollama_files = sorted(glob.glob('quick_benchmark_*.json'), reverse=True)
nanollm_files = sorted(glob.glob('nanollm_benchmark_*.json'), reverse=True)
eval_files = sorted(glob.glob('gemini_evaluations_*.json'), reverse=True)

print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

# Performance Results
print("\n📊 PERFORMANCE METRICS")
print("-" * 70)

if ollama_files:
    with open(ollama_files[0]) as f:
        ollama_data = json.load(f)
    
    print("\n[OLLAMA]")
    models = {}
    for r in ollama_data:
        model = r['model']
        if model not in models:
            models[model] = []
        models[model].append(r)
    
    for model, results in models.items():
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        avg_time = sum(r['total_time'] for r in results) / len(results)
        print(f"  {model}:")
        print(f"    Throughput: {avg_throughput:.1f} tokens/s")
        print(f"    Avg Time:   {avg_time:.1f}s")

if nanollm_files:
    with open(nanollm_files[0]) as f:
        nanollm_data = json.load(f)
    
    print("\n[NANOLLM]")
    models = {}
    for r in nanollm_data:
        model = r['model']
        if model not in models:
            models[model] = []
        models[model].append(r)
    
    for model, results in models.items():
        avg_throughput = sum(r['throughput'] for r in results) / len(results)
        avg_time = sum(r['total_time'] for r in results) / len(results)
        print(f"  {model}:")
        print(f"    Throughput: {avg_throughput:.1f} tokens/s")
        print(f"    Avg Time:   {avg_time:.1f}s")

# Quality Results
if eval_files:
    print("\n\n⭐ QUALITY SCORES (Gemini Judge)")
    print("-" * 70)
    
    with open(eval_files[0]) as f:
        eval_data = json.load(f)
    
    backends = {}
    for ev in eval_data:
        backend = ev.get('backend', 'unknown')
        model = ev['model']
        key = f"{backend}/{model}"
        
        if key not in backends:
            backends[key] = []
        backends[key].append(ev['average'])
    
    for key in sorted(backends.keys()):
        scores = backends[key]
        avg = sum(scores) / len(scores)
        backend, model = key.split('/', 1)
        print(f"\n[{backend.upper()}] {model}:")
        print(f"  Average Quality: {avg:.2f}/10")
        
        # Find detailed scores for this model
        model_evals = [e for e in eval_data if e['model'] == model and e.get('backend') == backend]
        if model_evals:
            avg_correctness = sum(e['correctness'] for e in model_evals) / len(model_evals)
            avg_relevance = sum(e['relevance'] for e in model_evals) / len(model_evals)
            avg_completeness = sum(e['completeness'] for e in model_evals) / len(model_evals)
            avg_coherence = sum(e['coherence'] for e in model_evals) / len(model_evals)
            avg_reasoning = sum(e['reasoning_quality'] for e in model_evals) / len(model_evals)
            
            print(f"    - Correctness:  {avg_correctness:.1f}/10")
            print(f"    - Relevance:    {avg_relevance:.1f}/10")
            print(f"    - Completeness: {avg_completeness:.1f}/10")
            print(f"    - Coherence:    {avg_coherence:.1f}/10")
            print(f"    - Reasoning:    {avg_reasoning:.1f}/10")

print("\n" + "=" * 70)
print("\n📁 GENERATED FILES:")
if ollama_files:
    print(f"  Ollama:     {ollama_files[0]}")
if nanollm_files:
    print(f"  NanoLLM:    {nanollm_files[0]}")
if eval_files:
    print(f"  Evaluation: {eval_files[0]}")

print("\n" + "=" * 70)
print("✅ Benchmark suite complete!")
print("=" * 70)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Summary report generated${NC}"
else
    echo -e "${YELLOW}⚠ Could not generate summary report${NC}"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}   All benchmarks complete!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "Next steps:"
echo "  - Review results above"
echo "  - Check plots/ directory for visualizations"
echo "  - Examine JSON files for detailed data"
echo ""
