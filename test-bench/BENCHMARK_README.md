# LLM Benchmark Suite

Comprehensive benchmarking tool for comparing NanoLLM and Ollama performance on Jetson hardware.

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
