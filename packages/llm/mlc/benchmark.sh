#!/usr/bin/env bash
#
# Llama benchmark with MLC. This script should be invoked from the host and will run 
# the MLC container with the commands to benchmark locally available models.
# It will add its collected performance data to jetson-containers/data/benchmarks/mlc.csv 
#
# Models should be available locally at /data/models/mlc/${MLC_VERSION}/${model_name}-${QUANTIZATION}-MLC
# You can run it like this:
#
#    ./benchmark.sh meta-llama/Llama-2-7b-hf
#
# If a model is not specified, then the default set of models will be benchmarked.
# See the environment variables below and their defaults for model settings to change.
#
# These are the possible quantization methods that can be set like QUANTIZATION=q4f16_ft
#
#  (MLC 0.1.0) q4f16_0,q4f16_1,q4f16_2,q4f16_ft,q4f16_ft_group,q4f32_0,q4f32_1,q8f16_ft,q8f16_ft_group,q8f16_1
#  (MLC 0.1.1) q4f16_0,q4f16_1,q4f32_1,q4f16_2,q4f16_autoawq,q4f16_ft,e5m2_e5m2_f16
#
set -ex

: "${MLC_VERSION:=0.1.4}"

: "${QUANTIZATION:=q4f16_ft}"
: "${SKIP_QUANTIZATION:=no}"
: "${USE_SAFETENSORS:=yes}"

#: "${MAX_CONTEXT_LEN:=4096}"
: "${MAX_NUM_PROMPTS:=4}"
: "${CONV_TEMPLATE:=llama-2}"
: "${PROMPT:=/data/prompts/completion_16.json}"

: "${OUTPUT_CSV:=/data/benchmarks/mlc.csv}"


function benchmark() 
{
    local model_repo=$1
    local model_name=$(basename $model_repo)
    local model_root="/data/models/mlc/${MLC_VERSION}"
    
    # Use local model path instead of downloading from HuggingFace
    local local_model_path="${model_root}/${model_name}-${QUANTIZATION}-MLC"
    
    if [ ${MLC_VERSION:4} -ge 4 ]; then
      mkdir -p $(jetson-containers data)/models/mlc/cache || true ;
      
      run_cmd="\
        python3 benchmark.py \
          --model ${local_model_path} \
          --max-new-tokens 128 \
          --max-num-prompts 4 \
          --prompt $PROMPT \
          --save ${OUTPUT_CSV} "
        
      if [ -n "$MAX_CONTEXT_LEN" ]; then
        run_cmd="$run_cmd --max-context-len $MAX_CONTEXT_LEN"
      fi
      
      if [ -n "$PREFILL_CHUNK_SIZE" ]; then
        run_cmd="$run_cmd --prefill-chunk-size $PREFILL_CHUNK_SIZE"
      fi
      
      run_cmd="$run_cmd ; rm -rf /data/models/mlc/cache/* || true ; "
    else
      # For older MLC versions, use local model directory directly
      run_cmd="\
        bash test.sh $model_name ${local_model_path} "
    fi
    
    jetson-containers run \
        -e QUANTIZATION=${QUANTIZATION} \
        -e SKIP_QUANTIZATION=${SKIP_QUANTIZATION} \
        -e USE_SAFETENSORS=${USE_SAFETENSORS} \
        -e MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN} \
        -e MAX_NUM_PROMPTS=${MAX_NUM_PROMPTS} \
        -e CONV_TEMPLATE=${CONV_TEMPLATE} \
        -e PROMPT=${PROMPT} \
        -e OUTPUT_CSV=${OUTPUT_CSV} \
        -e MODEL_REPO=${model_repo} \
        -e MODEL_ROOT=${model_root} \
        -v $(jetson-containers root)/packages/llm/mlc:/test \
        -w /test \
        dustynv/mlc:0.1.4-r36.4.2 /bin/bash -c "$run_cmd"
}
            
   
if [ "$#" -gt 0 ]; then
    benchmark "$@"
    exit 0 
fi


#MLC_VERSION="0.1.0" MAX_CONTEXT_LEN=4096 USE_SAFETENSORS=off benchmark "princeton-nlp/Sheared-LLaMA-1.3B"
#MLC_VERSION="0.1.0" MAX_CONTEXT_LEN=4096 USE_SAFETENSORS=off benchmark "princeton-nlp/Sheared-LLaMA-2.7B"

#MLC_VERSION="0.1.0" MAX_CONTEXT_LEN=4096 benchmark "meta-llama/Llama-2-7b-hf"
#MLC_VERSION="0.1.1" MAX_CONTEXT_LEN=8192 benchmark "meta-llama/Meta-Llama-3-8B"

benchmark "meta-llama/Llama-3.2-1B-Instruct"
benchmark "meta-llama/Llama-3.2-3B-Instruct"
benchmark "meta-llama/Llama-3.1-8B-Instruct"
benchmark "meta-llama/Llama-2-7b-chat-hf"

MAX_CONTEXT_LEN=4096 PREFILL_CHUNK_SIZE=4096 benchmark "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CONTEXT_LEN=4096 PREFILL_CHUNK_SIZE=4096 benchmark "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CONTEXT_LEN=2048 PREFILL_CHUNK_SIZE=1024 benchmark "Qwen/Qwen2.5-7B-Instruct"

QUANTIZATION="q4f16_1" benchmark "google/gemma-2-2b-it"
#QUANTIZATION="q4f16_1" benchmark "google/gemma-2-9b-it"

benchmark "microsoft/Phi-3.5-mini-instruct"

benchmark "HuggingFaceTB/SmolLM2-135M-Instruct"
benchmark "HuggingFaceTB/SmolLM2-360M-Instruct"
benchmark "HuggingFaceTB/SmolLM2-1.7B-Instruct"