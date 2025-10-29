# LLM Benchmark Suite for Jetson

Complete benchmarking system for comparing Ollama and NanoLLM performance with AI-powered quality evaluation.

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)
```bash
cd /home/nvidia/Developer/vision-language-model-testbench/jetson-containers/test-bench
source ~/.config/benchmark/gemini_key.sh  # Load API key
./run_all_benchmarks.sh
```

This will:
- Benchmark 3 models on Ollama
- Benchmark 3 models on NanoLLM  
- Evaluate all responses with Gemini AI
- Generate comprehensive comparison report

### Option 2: Skip NanoLLM (Faster - 5 minutes)
```bash
SKIP_NANOLLM=yes ./run_all_benchmarks.sh
```

### Option 3: Individual Components
```bash
# Just Ollama
python3 quick_benchmark.py

# Just NanoLLM
python3 benchmark_nanollm.py

# Just evaluation
python3 evaluate_with_gemini.py
```

---

## 📋 Prerequisites

### Required

1. **Gemini API Key** (for quality evaluation)
```bash
# Set up once
export GEMINI_API_KEY='your-api-key-here'
mkdir -p ~/.config/benchmark
echo "export GEMINI_API_KEY='your-api-key-here'" > ~/.config/benchmark/gemini_key.sh
chmod 600 ~/.config/benchmark/gemini_key.sh

# Load for each session
source ~/.config/benchmark/gemini_key.sh
```

2. **Ollama Running** (for Ollama benchmarks)
```bash
sudo systemctl start ollama
curl http://localhost:11434/api/tags  # Verify
```

3. **jetson-containers** (for NanoLLM benchmarks)
```bash
which jetson-containers  # Should return path
```

### Optional
- Google Generative AI Python package (auto-installed)
- HuggingFace token (for NanoLLM gated models)

---

## 🎯 Available Scripts

### Master Script
**`run_all_benchmarks.sh`** - Complete benchmark pipeline
- Runs Ollama and NanoLLM benchmarks
- Evaluates with Gemini
- Generates summary report

Environment variables:
- `SKIP_OLLAMA=yes` - Skip Ollama benchmarks
- `SKIP_NANOLLM=yes` - Skip NanoLLM benchmarks  
- `SKIP_EVALUATION=yes` - Skip quality evaluation

### Benchmark Scripts

**`quick_benchmark.py`** - Fast Ollama benchmark
- Tests available Ollama models
- 3 prompts: reasoning, code, summarization
- Output: `quick_benchmark_YYYYMMDD_HHMMSS.json`

**`benchmark_nanollm.py`** - NanoLLM benchmark via container
- Uses `nano_llm.chat` API with MLC backend
- Same 3 prompts for fair comparison
- Output: `nanollm_benchmark_YYYYMMDD_HHMMSS.json`

**`benchmark_llm.py`** - Full comprehensive benchmark
- Detailed metrics: GPU, CPU, memory usage
- Multiple prompts and models
- Generates visualizations and CSV
- Output: Multiple files + `plots/` directory

### Evaluation Scripts

**`evaluate_with_gemini.py`** - AI-powered quality scoring
- Uses Gemini 2.5 Flash as judge
- Evaluates on 5 criteria (correctness, relevance, completeness, coherence, reasoning)
- Handles both Ollama and NanoLLM results
- Output: `gemini_evaluations_combined_YYYYMMDD_HHMMSS.json`

### Utility Scripts

**`check_benchmark_setup.py`** - System verification
- Checks Python version and dependencies
- Verifies Ollama/NanoLLM availability
- Confirms tegrastats for GPU monitoring

---

## 📊 Output Files

### Performance Data
- `quick_benchmark_*.json` - Ollama performance results
- `nanollm_benchmark_*.json` - NanoLLM performance results
- `benchmark_results.json` - Full benchmark raw data
- `results_summary.csv` - Aggregated statistics

### Quality Scores
- `gemini_evaluations_*.json` - Individual evaluation results
- `gemini_evaluations_combined_*.json` - Combined Ollama + NanoLLM scores
- `quality_scores.json` - Full benchmark quality data

### Visualizations (from full benchmark)
- `plots/performance_comparison.png` - Throughput comparison
- `plots/first_token_latency.png` - Latency analysis
- `plots/quality_radar.png` - Quality breakdown by criteria
- `plots/average_quality.png` - Overall quality scores

---

## 🔧 Configuration

### Models Tested

**Ollama** (edit in `quick_benchmark.py`):
```python
MODELS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "gemma3:4b",
]
```

**NanoLLM** (edit in `benchmark_nanollm.py`):
```python
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
```

### Test Prompts

Edit `PROMPTS` list in any benchmark script:
```python
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
```

### NanoLLM Settings

In `benchmark_nanollm.py`:
```python
CONTAINER = "dustynv/nano_llm:r36.4.0"
QUANTIZATION = "q4f16_ft"  # Options: q4f16_ft, q8f16_ft, q2_K, etc.
MAX_TOKENS = 500  # Max output tokens
```

---

## 📈 Expected Results

### Performance Metrics
```
[OLLAMA] llama3.2:3b
  Throughput: 18.9 tokens/s
  Avg Time:   14.3s

[NANOLLM] Llama-3.2-3B-Instruct
  Throughput: XX.X tokens/s
  Avg Time:   XX.Xs
```

### Quality Scores (Gemini Judge)
```
[OLLAMA] llama3.2:3b
  Average Quality: 9.33/10
  - Correctness:  9.7/10
  - Relevance:    8.3/10
  - Completeness: 9.3/10
  - Coherence:    10.0/10
  - Reasoning:    9.3/10
```

---

## ⏱️ Timing Estimates

| Task | Duration | Notes |
|------|----------|-------|
| Ollama (3 models × 3 prompts) | 2-5 min | Fast with local inference |
| NanoLLM (3 models × 3 prompts) | 15-30 min | MLC compilation overhead |
| Gemini Evaluation | 1-2 min | Cloud API (fast) |
| **Total (all)** | **20-40 min** | First run |
| **Ollama only** | **5-10 min** | Quick comparison |

---

## 🐛 Troubleshooting

### Ollama Issues
```bash
# Start Ollama
sudo systemctl start ollama

# Check status
curl http://localhost:11434/api/tags

# List available models
ollama list
```

### NanoLLM Issues
```bash
# Check container is available
docker images | grep nano_llm

# Verify jetson-containers
which jetson-containers

# Skip NanoLLM if models need compilation
SKIP_NANOLLM=yes ./run_all_benchmarks.sh
```

### Gemini API Issues
```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Reload from config
source ~/.config/benchmark/gemini_key.sh

# Test API
python3 -c "import google.generativeai as genai; genai.configure(api_key='$GEMINI_API_KEY'); print('✓ OK')"
```

### Permission Errors
```bash
# Make scripts executable
chmod +x *.sh *.py

# Check current directory
pwd  # Should be: .../jetson-containers/test-bench
```

---

## 📁 Directory Structure

```
test-bench/
├── run_all_benchmarks.sh          ⭐ Master script
├── quick_benchmark.py             📊 Ollama benchmark
├── benchmark_nanollm.py           📊 NanoLLM benchmark  
├── benchmark_llm.py               📊 Full benchmark suite
├── evaluate_with_gemini.py        🤖 Gemini evaluation
├── check_benchmark_setup.py       🔧 System check
│
├── quick_benchmark_*.json         📄 Ollama results
├── nanollm_benchmark_*.json       📄 NanoLLM results
├── gemini_evaluations_*.json      📄 Quality scores
│
└── plots/                         📈 Visualizations (if generated)
```

---

## 🎓 Usage Examples

### Basic Workflow
```bash
cd /home/nvidia/Developer/vision-language-model-testbench/jetson-containers/test-bench

# 1. Verify system
python3 check_benchmark_setup.py

# 2. Load API key
source ~/.config/benchmark/gemini_key.sh

# 3. Run benchmarks
./run_all_benchmarks.sh

# 4. Review results (printed to console)
```

### Quick Test (Ollama Only)
```bash
SKIP_NANOLLM=yes ./run_all_benchmarks.sh
```

### Custom Model Test
```bash
# Edit quick_benchmark.py to test specific models
# Then run:
python3 quick_benchmark.py
python3 evaluate_with_gemini.py
```

### Background Execution
```bash
nohup ./run_all_benchmarks.sh > benchmark_full.log 2>&1 &

# Monitor progress
tail -f benchmark_full.log

# Check if still running
ps aux | grep benchmark
```

---

## 📞 Support

### Check Logs
```bash
# View recent benchmark output
cat quick_benchmark_*.json | jq '.[0]'

# Check evaluation results
cat gemini_evaluations_*.json | jq '.[0]'
```

### Verify Installation
```bash
python3 check_benchmark_setup.py
```

### Clean Old Results
```bash
# Remove old benchmark files (optional)
rm quick_benchmark_*.json
rm nanollm_benchmark_*.json
rm gemini_evaluations_*.json
```

---

## 🔑 Key Features

✅ **Dual Backend Comparison** - Compare Ollama vs NanoLLM side-by-side  
✅ **AI-Powered Evaluation** - Gemini 2.5 Flash judges response quality  
✅ **Comprehensive Metrics** - Throughput, latency, GPU/CPU usage  
✅ **Easy to Use** - Single command for complete benchmark  
✅ **Flexible** - Skip components, test custom models/prompts  
✅ **Production Ready** - Handles errors, provides detailed logs  

---

**Ready to benchmark?** Run: `./run_all_benchmarks.sh`

## Features

- **Multi-Model Benchmarking**: Tests Llama-3.2-1B, 3B, and 3.1-8B models
- **Dual Backend Support**: NanoLLM and Ollama comparison
- **Comprehensive Metrics**:
  - Total latency
  - First-token latency
  - Tokens per second throughput
  - GPU/CPU usage via tegrastats
  - Memory consumption
  - Model load time
- **LLM-as-Judge Evaluation**: Quality scoring on 5 dimensions
- **Rich Visualizations**: Performance charts and quality radar plots

## Prerequisites

### System Requirements
- Python 3.10+
- NanoLLM server running on `localhost:8080` (optional)
- Ollama server running on `localhost:11434` (optional)
- At least one backend must be running

### Python Dependencies
```bash
pip3 install --user openai pandas matplotlib requests psutil
```

### Optional: OpenAI API for Judge
Set environment variable for GPT-4o-mini judge (otherwise uses local Ollama):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Run
```bash
cd /home/nvidia/Developer/vision-language-model-testbench
python3 benchmark_llm.py
```

### Check Server Status First
The script automatically checks if NanoLLM and Ollama are running and will benchmark on available backends.

### Starting Servers

#### NanoLLM
```bash
# Start NanoLLM server
jetson-containers run --name nanollm dustynv/nanollm:latest \
  python3 -m nanollm.server --port 8080
```

#### Ollama
```bash
# Start Ollama server
ollama serve
```

## Benchmark Process

The script will:

1. **Check server availability** (NanoLLM and/or Ollama)
2. **Run 7 prompts** across 3 models on each available backend
3. **Collect performance metrics** for each run
4. **Evaluate response quality** using LLM-as-Judge
5. **Generate outputs**:
   - `benchmark_results.json` - Raw performance data
   - `quality_scores.json` - LLM judge evaluations
   - `results_summary.csv` - Aggregated statistics
   - `plots/` directory with visualizations

## Output Files

### benchmark_results.json
Contains detailed metrics for each benchmark run:
```json
{
  "timestamp": "2025-10-28T10:30:45",
  "backend": "nanollm",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "prompt_id": "reasoning_1",
  "total_latency_s": 2.45,
  "first_token_latency_s": 0.15,
  "tokens_generated": 128,
  "tokens_per_sec": 52.24,
  "gpu_usage_start": 5,
  "gpu_usage_end": 85,
  ...
}
```

### quality_scores.json
LLM-as-Judge evaluations (1-10 scale):
```json
{
  "backend": "ollama",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "correctness": 9,
  "relevance": 8,
  "completeness": 8,
  "coherence": 9,
  "reasoning_quality": 8,
  "average_quality": 8.4
}
```

### results_summary.csv
Aggregated statistics with mean and standard deviation for all metrics.

### plots/
- `performance_comparison.png` - Latency and throughput comparison
- `first_token_latency.png` - Time to first token
- `quality_radar.png` - Quality metrics radar chart
- `average_quality.png` - Overall quality scores

## Test Prompts

The benchmark includes 7 diverse prompts:

1. **Reasoning**: Recursion explanation
2. **Code**: Palindrome checker function
3. **Summarization**: OOP principles
4. **Reasoning**: Train journey calculation
5. **Code**: Two-sum algorithm
6. **Summarization**: ML vs DL explanation
7. **Reasoning**: AI ethics in healthcare

## Customization

### Add More Prompts
Edit the `PROMPTS` list in `benchmark_llm.py`:
```python
PROMPTS = [
    {
        "id": "custom_1",
        "category": "reasoning",
        "prompt": "Your prompt here..."
    },
    ...
]
```

### Test Different Models
Edit the `MODELS` list:
```python
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "your-model/here",
    ...
]
```

### Change Server URLs
Modify in `__init__`:
```python
self.nanollm_url = "http://localhost:8080"
self.ollama_url = "http://localhost:11434"
```

## Troubleshooting

### "No servers are running"
- Ensure NanoLLM or Ollama is started
- Check server URLs are correct
- Verify ports are not blocked

### "Quality evaluation failed"
- Install OpenAI: `pip3 install openai`
- Or ensure local Ollama judge model is available: `ollama pull llama3.2:3b-instruct`

### GPU stats showing 0%
- Script falls back to psutil if tegrastats unavailable
- This is normal on non-Jetson systems

### Import errors
```bash
pip3 install --user openai pandas matplotlib requests psutil
```

## Performance Tips

- **Close other GPU applications** for accurate measurements
- **Run multiple times** for statistical significance
- **Monitor system** with `jtop` during benchmarking
- **Warm up models** - first run may be slower

## Example Summary Output

```
NanoLLM - Llama-3.2-1B-Instruct
------------------------------------------------------------
  Performance Metrics:
    Latency:          2.45s ± 0.23s
    First Token:      0.152s ± 0.018s
    Throughput:       52.24 ± 4.12 tokens/s
    Tokens Generated: 128 ± 15
  Resource Usage:
    GPU Usage:        85.2% ± 5.3%
    CPU Usage:        45.1% ± 3.2%
  Quality Scores (1-10):
    Correctness:      8.5 ± 0.7
    Relevance:        8.8 ± 0.5
    Completeness:     8.2 ± 0.6
    Coherence:        8.9 ± 0.4
    Reasoning:        8.4 ± 0.8
    Average:          8.6 ± 0.6
```

## License

This benchmark tool is provided as-is for research and evaluation purposes.
