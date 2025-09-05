import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def mirostat(model, tokenizer, prompt, max_length=50, device='cpu', temperature=1.0, target_ce=3.0, learning_rate=0.1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mu = 2 * target_ce  # Initial mu value / "maximal surprisal"

    # TODO: YOUR CODE HERE -- additional variable init
    # We will not be checking this section for correctness,
    # But you will probably eventually want to set up some 
    # extra variables here for plotting metrics.
    # Our advice is to fill out the other sections first!

    for step in range(max_length):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            
            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending = True)
        
        # TODO: YOUR CODE HERE -- Estimate Zipf's exponent
        # Following Basu et al, use m=100 (i.e. use only the top 100 tokens(' diffs) to estimate the exponent)
        # Refer to Equation 30 https://arxiv.org/pdf/2007.14966#equation.C.30 for pointers
        
        s_hat = None # replace with your own expression

        # TODO: YOUR CODE HERE -- Compute k using Zipf exponent
        k = None

        # top k sampling
        topk_logits = sorted_logits[0:k]
        topk_inds = sorted_inds[0:k]
        topk_probs = torch.softmax(topk_logits, dim=1)
        next_tok = topk_inds[0, torch.multinomial(topk_probs, num_samples=1)]
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

        # TODO: YOUR CODE HERE -- Compute surprisal error and adjust mu accordingly
        err = None
        mu = None
        
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "Once upon a time,"
    result = mirostat(model, tokenizer, prompt, max_length=256, device=device, temperature=1.0, target_ce=3.0, learning_rate=0.1)
    print(result)