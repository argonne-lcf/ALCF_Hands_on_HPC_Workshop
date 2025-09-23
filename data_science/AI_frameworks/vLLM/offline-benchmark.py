import torch 
# import intel_extension_for_pytorch as ipex

import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--max-input-length", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--tensor-parallel", type=int, default=1)
parser.add_argument("--max-model-len", type=int, default=2048)
parser.add_argument('--n',
                    type=int,
                    default=1,
                    help='Number of generated sequences per prompt.')
args = parser.parse_args()

from vllm import SamplingParams, LLM


def generate_input(args):
    random_words = ["France" for _ in range(args.max_input_length)]

    input_id = ""

    for word in random_words:
        input_id = input_id + word + " "

    input_id = input_id[:-1]

    input_list = []

    for batch_size in range(args.batch_size):
        input_list.append(input_id)

    return input_list

prompts = generate_input(args)

# Create a sampling params object.
sampling_params = SamplingParams(
    n=args.n,
    temperature=0.0, #if args.use_beam_search else 1.0,
    top_p=1.0,
    ignore_eos=True,
    max_tokens=args.max_new_tokens,
)
print(sampling_params)

# Create an LLM.

llm = LLM(model=args.model,

          device="xpu",
          dtype="float16",
          enforce_eager=True,
          tensor_parallel_size=args.tensor_parallel,
          gpu_memory_utilization=0.6,
          trust_remote_code=True,
          distributed_executor_backend="ray",
          max_model_len=args.max_model_len)


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.


start_time = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
end_time = time.perf_counter()
latency = end_time - start_time
total_num_tokens = args.batch_size*(args.max_input_length + args.max_new_tokens)
print("Total Number of Tokens = ", total_num_tokens)
throughput = total_num_tokens/latency
print("Throughput = ", throughput)


