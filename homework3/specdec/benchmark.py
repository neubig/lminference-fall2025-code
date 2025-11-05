"""   
# Benchmarking Script for Speculative Decoding
Gives empirical speedup measurements for speculative decoding vs standard
autoregressive decoding on a set of prompts.

Usage:
    CUDA_VISIBLE_DEVICES=0 python specdec/benchmark.py --target Qwen/Qwen3-8B --draft Qwen/Qwen3-0.6B --lookahead 3 --runs_per_prompt 2
    # should take around 4 mins to run

# To get a better understanding of performance, it's useful to analyze
# these results at the dataset (task)/ per query level.
"""
import argparse
import torch
import json
import numpy as np
import os
from specdec import SpeculativeDecoder, SpecDecSamplingConfig


def get_args():
    p = argparse.ArgumentParser(description="Speculative Decoding Benchmark")
    p.add_argument("--target", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--draft", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lookahead", type=int, default=3)
    p.add_argument("--runs_per_prompt", type=int, default=1)
    p.add_argument("--prompts", default="specdec/prompts.jsonl")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(42)

    # --- 1. Configuration ---
    config = SpecDecSamplingConfig(
        target_name=args.target,
        draft_name=args.draft,
    )
    config.max_new_tokens = 50
    config.temperature = 0.1
    config.dtype = torch.bfloat16
    config.debug = args.debug
    config.device = args.device
    config.lookahead_K = args.lookahead

    N_RUNS = args.runs_per_prompt  # Number of runs per prompt

    print("Initializing speculative decoder...")
    spec_decoder = SpeculativeDecoder(config)
    print("Decoder initialized.")

    # --- 2. Load Prompts ---
    prompts_file = args.prompts
    prompts = []

    with open(prompts_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if "formatted_prompt" in data:
                prompts.append(data["formatted_prompt"])

    print(f"Loaded {len(prompts)} prompts from {prompts_file}.")

    if not prompts:
        print("Error: No prompts found. Exiting.")
        return

    # --- 3. Run Benchmark ---
    sd_times, ar_times, acceptance_rates = [], [], []

    # Warmup
    print("\n--- WARMUP ---")
    tokenized_prefix = spec_decoder.tokenizer("warmup", return_tensors="pt").input_ids.to(config.device)
    _ = spec_decoder.sd_sample(tokenized_prefix, max_new_tokens=10,
                               lookahead=config.lookahead_K, temperature=config.temperature)
    _ = spec_decoder.ar_sample(spec_decoder.target_model,
                               tokenized_prefix,
                               max_new_tokens=10, temperature=config.temperature)
    print("Warmup complete.")

    print(f"\nStarting benchmark with {len(prompts)} prompts...")
    for i, prefix in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)} ({N_RUNS} runs)...")

        tokenized_prefix = spec_decoder.tokenizer(prefix, return_tensors="pt").input_ids.to(config.device)
        prompt_sd_times, prompt_ar_times, prompt_acceptance_rates = [], [], []

        for j in range(N_RUNS):
            print(f"    Run {j+1}/{N_RUNS}...")

            # --- Generate with speculative sampling ---
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            sd_output, emp_ar = spec_decoder.sd_sample(
                tokenized_prompt=tokenized_prefix,
                max_new_tokens=config.max_new_tokens,
                lookahead=config.lookahead_K,
                temperature=config.temperature
            )
            end.record()
            torch.cuda.synchronize()
            sd_time = start.elapsed_time(end)
            prompt_sd_times.append(sd_time)
            prompt_acceptance_rates.append(emp_ar)

            # --- Generate with standard autoregressive ---
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            ar_output, ar_logits = spec_decoder.ar_sample(
                model=spec_decoder.target_model,
                tokenized_prompt=tokenized_prefix,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature
            )
            end_time.record()
            torch.cuda.synchronize()
            ar_time = start_time.elapsed_time(end_time)
            prompt_ar_times.append(ar_time)

        # Average per prompt
        sd_times.append(np.mean(prompt_sd_times))
        ar_times.append(np.mean(prompt_ar_times))
        acceptance_rates.append(np.mean(prompt_acceptance_rates))

    print("\nBenchmark finished.")

    # --- 4. Print & Save Summary ---
    avg_sd_time = np.mean(sd_times)
    avg_ar_time = np.mean(ar_times)
    avg_acceptance_rate = np.mean(acceptance_rates)
    avg_sd_time_s = avg_sd_time / 1000.0
    avg_ar_time_s = avg_ar_time / 1000.0
    avg_speedup = avg_ar_time / avg_sd_time

    summary = (
        "\n---(Benchmark Summary) ---\n"
        f"Processed {len(sd_times)} / {len(prompts)} prompts successfully.\n"
        f"Averaged over {N_RUNS} runs per prompt.\n"
        f"Generation params: max_new_tokens={config.max_new_tokens}, "
        f"lookahead={config.lookahead_K}, temp={config.temperature}\n"
        "---------------------------------\n"
        f"Average AR Time: {avg_ar_time_s:.4f}s\n"
        f"Average SD Time: {avg_sd_time_s:.4f}s\n"
        f"Average Acceptance Rate: {avg_acceptance_rate:.2%}\n"
        f"Average Empirical Speedup: {avg_speedup:.2f}x\n"
        "---------------------------------\n"
    )

    print(summary)

    # --- 5. Save to log file ---
    os.makedirs("benchmark_dir", exist_ok=True)
    log_path = f"benchmark_dir/{args.target.replace('/', '_')}_{args.draft.replace('/', '_')}_overall_speedup.log"
    with open(log_path, "w") as fout:
        fout.write(summary)

    print(f"Results written to {log_path}")


if __name__ == "__main__":
    main()