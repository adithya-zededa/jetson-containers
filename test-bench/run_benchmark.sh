#!/bin/bash
# Automated LLM Benchmark Execution Script

set -e

echo "================================================================"
echo "LLM Benchmark Suite - Automated Execution"
echo "================================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Step 1: Check system readiness
echo "Step 1: Checking system setup..."
python3 check_benchmark_setup.py

# Ask user if they want to continue
echo ""
read -p "Continue with benchmarking? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled."
    exit 0
fi

# Step 2: Ask for test run or full run
echo ""
echo "Choose benchmark mode:"
echo "  1) Quick test (1 model, 1 prompt)"
echo "  2) Full benchmark (3 models, 7 prompts)"
echo ""
read -p "Enter choice (1 or 2): " -n 1 -r
echo ""

if [[ $REPLY == "1" ]]; then
    echo ""
    echo "Running quick test..."
    python3 test_benchmark.py
elif [[ $REPLY == "2" ]]; then
    echo ""
    echo "Running full benchmark..."
    echo "This may take 30-60 minutes depending on your system."
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 benchmark_llm.py
    else
        echo "Benchmark cancelled."
        exit 0
    fi
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Step 3: Display results
echo ""
echo "================================================================"
echo "Benchmark Complete!"
echo "================================================================"
echo ""
echo "Generated files:"
ls -lh benchmark_results.json quality_scores.json results_summary.csv 2>/dev/null || echo "  (Check for errors above)"
echo ""
echo "Plots:"
ls -lh plots/*.png 2>/dev/null || echo "  (No plots generated)"
echo ""
echo "View summary:"
echo "  cat results_summary.csv"
echo ""
echo "View detailed results:"
echo "  python3 -m json.tool benchmark_results.json | less"
echo ""
echo "================================================================"
