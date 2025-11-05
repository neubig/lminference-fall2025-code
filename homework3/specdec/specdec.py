"""
Speculative Decoding Implementation

Reference Papers:
1. Fast Inference from Transformers via Speculative Decoding (https://arxiv.org/pdf/2211.17192)
2. Accelerating Large Language Model Decoding with Speculative Sampling (https://arxiv.org/pdf/2302.01318)

This implementation follows Algorithm 2 from Paper 2 (DeepMind).
See Theorem 1 for why the rejection sampling preserves the target distribution.
"""
import torch
import transformers
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored

torch.manual_seed(42)
transformers.set_seed(42)


class SamplingConfig:
    def __init__(self,
                 max_new_tokens: int=50,
                 temperature: float=1.0,
                 lookahead_K: int=3,
                 device: str = "cuda:0",
                 debug: bool = False):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lookahead_K = lookahead_K
        self.debug = debug
        self.dtype = torch.bfloat16


class SpecDecSamplingConfig(SamplingConfig):
    def __init__(self,
                 target_name: str,
                 draft_name: str):
        super().__init__()
        self.target_name = target_name
        self.draft_name = draft_name


class SpeculativeDecoder:
    def __init__(self, config: SpecDecSamplingConfig):
        """
        Initialize target model, draft model, and tokenizer.
        Set models to eval mode.
        """
        self.config = config
        self.device = config.device
        self.temperature = config.temperature
        
        # TODO: Load models and tokenizer
        raise NotImplementedError()
    
    def max_fn(self, x):
        """Max function from paper 2 (f)_+"""
        return NotImplementedError()
    
    def get_distribution(self, logits, temperature, epsilon=1e-8):
        """Get probability distribution from logits"""
        # Softmax with temperature
        if temperature <= epsilon:
            # Greedy decoding
        # temperature scaling 
        logits /= temperature
        # normalize
        return NotImplementedError()
    
    @torch.inference_mode()
    def ar_sample(self, model, tokenized_prompt, max_new_tokens, temperature=1.0):
        """
        Standard autoregressive sampling.
        Returns  generated sequence and temperature temp-normalized probs."""
        # TODO: Implement autoregressive generation
        raise NotImplementedError()

    @torch.inference_mode()
    # debug flags are left as is for easier debugging / seeing where outputs diverge
    def sd_sample(self, tokenized_prompt, max_new_tokens, lookahead, temperature):
        """
        Speculative decoding (Algorithm 2 from Paper 2).
        
        Args:
            tokenized_prompt: [batch_size, seq_len]
            max_new_tokens: Total tokens to generate
            lookahead: Number of speculative tokens (K)
            temperature: Sampling temperature
            
        Returns:
            generated_tokens: [batch_size, max_new_tokens]
            acceptance_rate: Fraction of draft tokens accepted
        """
        debug = self.config.debug
        bsz, n = tokenized_prompt.shape
        assert bsz == 1, 'Batch size should be 1'
        target_len = n + max_new_tokens

        # Metrics
        accepted_count = 0
        draft_token_num = 0
        n_orig = n
        
        while n < target_len:
            # HINT: you dont want to overshoot on max_new_tokens
            corrected_lookahead = None # TODO
            # TODO: Generate K draft tokens
            

            if debug:
                drafted_text = self.tokenizer.decode(draft_outputs[0], 
                                                     skip_special_tokens=False)
                print(colored(f"Possible continuations: {drafted_text}", 'blue', 'on_black'))
            
            
            # TODO: Run target model on draft sequence to verify
            # TODO: For each draft token, compute acceptance probability and accept/reject
            
            for t in range(corrected_lookahead):
                # accept loop
                if r < accept_prob:
                    if debug:
                        accepted_token = self.tokenizer.decode(draft_token)
                        print(f"Accepted token: '{accepted_token}'")

                # reject loop
                else:
                    # TODO: Reject and resample from adjusted distribution
                    if debug:
                        rejected_token = self.tokenizer.decode(draft_token)
                        new_token_text = self.tokenizer.decode(new_token[0])
                        print(colored(f"Rejected: {rejected_token}", 'red', 'on_black'))
                        print(colored(f"Replaced with: {new_token_text}", 'green', 'on_black'))
                    break
            
            # TODO: Sample bonus token if all accepted
                
        return NotImplementedError()
