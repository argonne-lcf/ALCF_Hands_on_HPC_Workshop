from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
import argparse
import time

parser = argparse.ArgumentParser(description="vLLM Inference")
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B')
parser.add_argument('--speculative-model', type=str, default=None)
parser.add_argument('--num-speculative-tokens', type=int, default=None)
parser.add_argument('--speculative-draft-tensor-parallel-size', '-spec-draft-tp', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--quantization', '-q', choices=[*QUANTIZATION_METHODS, None],default=None)
parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=4)
parser.add_argument('--pipeline-parallel-size', '-pp', type=int, default=1)
parser.add_argument('--input-len', type=int, default=32)
parser.add_argument('--output-len', type=int, default=64)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--n', type=int, default=1, help='Number of generated sequences per prompt.')
parser.add_argument('--use-beam-search', action='store_true')
parser.add_argument('--trust-remote-code', action='store_true', help='trust remote code from huggingface')
parser.add_argument('--max-model-len', type=int, default=None, help='Maximum length of a sequence (including prompt and output). '
                    'If None, will be derived from the model.')
parser.add_argument('--dtype', type=str, default='auto',
                    choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
                    help='data type for model weights and activations. '
                    'The "auto" option will use FP16 precision '
                    'for FP32 and FP16 models, and BF16 precision '
                    'for BF16 models.')
parser.add_argument('--enforce-eager', action='store_true', help='enforce eager mode and disable CUDA graph')
parser.add_argument('--kv-cache-dtype', type=str, choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'], default="auto",
                    help='Data type for kv cache storage. If "auto", will use model '
                    'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
                    'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
parser.add_argument('--quantization-param-path', type=str, default=None,
                    help='Path to the JSON file containing the KV cache scaling factors. '
                    'This should generally be supplied, when KV cache dtype is FP8. '
                    'Otherwise, KV cache scaling factors default to 1.0, which may cause '
                    'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
                    'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
                    'instead supported for common inference criteria.')

parser.add_argument("--device", type=str, default="cuda", help='device type for vLLM execution')
parser.add_argument('--block-size', type=int, default=16, help='block size of key/value cache')
parser.add_argument('--enable-chunked-prefill', action='store_true', help='If True, the prefill requests can be chunked based on the '
                    'max_num_batched_tokens')
parser.add_argument("--enable-prefix-caching", action='store_true', help="Enable automatic prefix caching")
parser.add_argument('--download-dir', type=str, default=None, help='directory to download and load the weights, '
                    'default to the default cache dir of huggingface')
parser.add_argument('--gpu-memory-utilization', type=float, default=0.9, help='the fraction of GPU memory to be used for '
                    'the model executor, which can range from 0 to 1.'
                    'If unspecified, will use the default value of 0.9.')
args = parser.parse_args()

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.output_len)

llm = LLM(model=args.model,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_draft_tensor_parallel_size=args.speculative_draft_tensor_parallel_size,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        enable_chunked_prefill=args.enable_chunked_prefill,
        download_dir=args.download_dir,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
    ) 

start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()
latency = end_time - start_time

print(f"vLLM Generation Latency = {latency}")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")