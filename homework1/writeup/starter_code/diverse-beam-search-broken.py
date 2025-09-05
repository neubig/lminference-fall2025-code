import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def diverse_beam_search(model, tokenizer, prompt, beam_width=6, num_groups=3, max_length=50, device='cpu', diversity_penalty=1.0):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    group_size = beam_width // num_groups
    beams = [[(input_ids, 1.0)] for _ in range(num_groups)] # Each group has its own beam
    completed = [[] for _ in range(num_groups)]

    for step in range(max_length):
        for g in range(num_groups):
            new_beams = []
            prev_tokens_counts = {}
            # Collect tokens used by previous groups at this step for diversity penalty
            for prev_g in range(g):
                if beams[prev_g]:
                    for b in range(group_size):
                        if beams[prev_g][b][0][0, -1] == tokenizer.eos_token_id:
                            continue # if already reached EOS, skip
                        prev_tok = beams[prev_g][b][0][0, -1].item()
                        if prev_tok not in prev_tokens_counts:
                            prev_tokens_counts[prev_tok] = 0
                        prev_tokens_counts[prev_tok] += 1
            for seq, score in beams[g]:
                with torch.no_grad():
                    outputs = model(seq)
                    logits = outputs.logits[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                # Penalize tokens used by previous groups
                penalized_probs = probs.clone()
                for token in prev_tokens_counts:
                    penalized_probs[0, token] /= (diversity_penalty * prev_tokens_counts[token])
                topk_probs, topk_ids = torch.topk(probs, group_size)
                for i in range(group_size):
                    next_id = topk_ids[0, i].unsqueeze(0)
                    next_score = score * topk_probs[0, i].item()
                    new_seq = torch.cat([seq, next_id.unsqueeze(0)], dim=1)
                    if next_id == tokenizer.eos_token_id:
                        completed[g].append((new_seq, next_score))
                    else:
                        new_beams.append((new_seq, next_score))
            beams[g] = sorted(new_beams, key=lambda x: x[1] / len(x[0][0]), reverse=True)
        # Stop if all beams are empty
        if all(len(group) == 0 for group in beams):
            break

    # Gather completed beams
    all_completed = []
    for group in completed:
        all_completed.extend(group)
    for group in beams:
        all_completed.extend(group)
    all_completed = sorted(all_completed, key=lambda x: x[1] / len(x[0][0]), reverse=True)
    print(len(all_completed))
    return [tokenizer.decode(seq[0], skip_special_tokens=True) for seq, _ in all_completed[:beam_width]]

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "Once upon a time,"
    results = diverse_beam_search(model, tokenizer, prompt, beam_width=6, num_groups=3, max_length=50, device=device, diversity_penalty=2.0)
    for i, result in enumerate(results):
        print(f"Diverse Beam {i+1}: {result}")