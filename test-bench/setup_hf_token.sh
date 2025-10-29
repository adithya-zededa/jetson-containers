#!/usr/bin/env bash
#
# Setup HuggingFace authentication for NanoLLM benchmarking
#

echo "======================================"
echo "NanoLLM Benchmark Setup"
echo "======================================"
echo ""

# Check if token is already set
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "HuggingFace token not found in environment."
    echo ""
    echo "Please enter your HuggingFace token (it will not be displayed):"
    read -s HF_TOKEN
    export HUGGINGFACE_TOKEN=$HF_TOKEN
    echo ""
    echo "Token set for this session."
    echo ""
    echo "To make this permanent, add to your ~/.bashrc:"
    echo "  export HUGGINGFACE_TOKEN='your_token_here'"
else
    echo "✓ HuggingFace token found in environment"
fi

# Save to a local config file (git-ignored)
mkdir -p ~/.config/benchmark
echo "export HUGGINGFACE_TOKEN='${HUGGINGFACE_TOKEN}'" > ~/.config/benchmark/hf_token.sh
chmod 600 ~/.config/benchmark/hf_token.sh

echo ""
echo "✓ Token saved to ~/.config/benchmark/hf_token.sh"
echo ""
echo "To use in future sessions:"
echo "  source ~/.config/benchmark/hf_token.sh"
echo ""
