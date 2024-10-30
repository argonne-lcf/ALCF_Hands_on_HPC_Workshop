
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse

def batch_encode(prompts, tokenizer, prompt_len=512):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding="max_length", max_length=prompt_len)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    return input_tokens

def generate_prompt(model, tokenizer, prompts):
    input_tokens = batch_encode(prompts, tokenizer)
    generate_kwargs = dict(max_new_tokens=64, do_sample=False)
    output_ids = model.generate(**input_tokens, **generate_kwargs)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="vLLM Inference")
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype = torch.float16, device_map  = 'auto')
    model.seqlen = 2048
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]

    start_time = time.perf_counter()
    output = generate_prompt(model, tokenizer, prompts=prompts)
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    print(f"HuggingFace Generation Latency = {latency}")
    print(output)