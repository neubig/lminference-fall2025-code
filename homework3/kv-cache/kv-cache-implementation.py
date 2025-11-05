import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

import random, math, time
from tqdm import tqdm

# adapted from nanoGPT (https://github.com/karpathy/nanoGPT/tree/master)


class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, cache=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
        # TODO: ADD KV CACHE LOGIC HERE (3-5 lines)
        # If cache is provided:
        #   - Extract past_k and past_v from cache
        #   - Concatenate(past_k, new_k) and do the same with v
        T_k = k.size(-2) # Need to get the new total sequence length from key

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # TODO: use this following line and comment out the original 
        # att = att.masked_fill(self.bias[:,:,T_k-T:T_k, :T_k] == 0, float('-inf')) # use this
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # comment me out

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        new_cache = {"k": k, "v": v} # this has already been modified for you: expect a dictionary format

        return y # , new_cache # TODO: uncomment this part when done with this function

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, prev_layer_kv=None):
        # TODO: self.attn takes in and returns a KV cache; use it
        attn_output = self.attn(self.ln_1(x)) # THIS IS THE ORIGINAL LINE; replace it with your new code
        x = x + attn_output # LEAVE THIS THE SAME
        x = x + self.mlp(self.ln_2(x))
        return x #, cur_layer_kv # TODO: Uncomment this part once the rest of the function is implemented


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 #2**16 #65536
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kv_caches=None): # past_kv_cache is a list of caches, one per layer
        device = idx.device
        b, t = idx.size()
        t_past = 0 # this will be overwritten if past_kv_caches exists
        # TODO: get sequence length from cache if it exists (just use one).
        
        # TODO: shift the positional embeddings to account for the past positions' sequence length, in the following two lines of code
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # TODO: if no kv cache, create a list of None (of length self.config.n_layer) to pass to the blocks

        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            x = block(x) # TODO: adjust this line to use your modified forward() function (and kv)
            # TODO: append to new_kv_caches

        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x)
        loss = None

        return logits, loss # , new_kv_caches # TODO: uncomment this part when done with the rest of the function
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # TODO: You will have to modify this function substantially to make it work! 
        # First, you will populate the entire initial KV cache via a separate prefill step
        # We're giving this to you, but you'll have to comment the next line and uncomment the following block
        idx_next = None # Placeholder, overwritten below and used instead of idx_cond
        """
        logits, _, kv_cache = self(idx)
        logits = logits[:, -1, :] / temperature # just the stuff after the prompt!

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) # Sample the first new token

        # Append new idx to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

        """
        # Now, some tweaks in the generation loop.
        for _ in range(max_new_tokens): # TODO: generate just one fewer than max_new_tokens since now we've already generated one (via the prefill block above)
            # idx_next has shape (B, 1)
            idx_cond = idx # TODO: just comment out. Originally had a block size safeguard but we aren't worrying about that
            logits, _ = self(idx_cond) # TODO: You know the drill by now (if not, do the other functions first)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


config = GPTConfig()
model = GPT.from_pretrained('gpt2-xl').to('cuda')
model.eval()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

sequences = tokenizer("The quick brown fox", return_tensors="pt")['input_ids'].to('cuda')

gen_len = 512

with torch.no_grad():
    start = time.time()
    torch.cuda.synchronize()
    print(tokenizer.batch_decode(model.generate(sequences, gen_len)))
    torch.cuda.synchronize()
    end = time.time()
    print("generation of", gen_len, "tokens took", end-start, "seconds")


# output for run as is on an L40S
"""
loading weights from pretrained gpt: gpt2-xl
forcing vocab_size=50257, block_size=1024, bias=True
number of parameters: 1555.97M
['The quick brown fox jumped through the window, much to the annoyance of the rest of the family. After opening the door, Enrique surveyed the crime scene with a tense gaze. No blood in the bedroom nor in the kitchen. The next room was slightly darker than the bedroom and had broken, blood-drenched wooden boards and a broken dish to the left side of the kitchen. The balcony smoldering finally caused everyone\'s eyes to light in panic. El Patio del Hechos was nice, nice enough for someone to run a couple of hits of grated nutmeg in the warm-up, but definitely not nice enough to burn a family! The warm-up ended, and neither Enrique nor Veronica washed their hands in the sink. Finally El Patio del Hechos returned to its normal frayed state. After another question from Enrique, Veronica lifted the sandbag 1928 and carefully put it against the rice cooker. The clan of, ah, Patios del Hechos, looked up from their cramped progress angrily, and then quietly returned to their tasks, moving above the debris. The problems that Big Jim and Jesse had been facing spilled onto the balcony that overlooked the rest of the kitchen, and new ones quickly appeared. There was the explosion in the kitchen and heavy fire, light smoke too. The family smelled the sweat and the sewer-spray from near our ovens. A basso profundo would have uttered the curse word "Cuella" in an instantly recognizable shrill sound, yet the black eyebrows persisted and the black face remained stark.Kevin and I, having long accumulated other quantities of evidence, began to move our own items, advised by Kevin\'s assurance of our tweaker founder\'s incapacity to make a coke-fueled ravioli. Already defective electrical connections dangled people from the air, and the pool-bed was milky white in places. It cast a pale shadow over what was actually 25-years-old, built out of excessive insulation, poor micro-structures and used repairs (dust mask and power plug pin installed, poor venting and ventilation). Kevin, sighing, screwed a small piece of brass fishing hook into a groove in a shuttered and defunct valve, and his harnessed hand now swung like an airborne sumo uchi. Gradually the orphanage complex toppled toward the clogged sewer pipes. A dirty loaf of bread bobbed from adjacent loaves with Yosemite romanticism.Jogger Emily only had her paddle paddle, but a well-made glove full of felt']
generation of 512 tokens took 21.308462381362915 seconds
"""
# vs with kv implementation:
"""
loading weights from pretrained gpt: gpt2-xl
forcing vocab_size=50257, block_size=1024, bias=True
number of parameters: 1555.97M
['The quick brown fox jumped quickly on top of the body and looked at the entire corpse, as if it was unimportant. "So not… roommate… but a dog you know…" Crabbe snorted as he stowed the dead girl. Moisture sprayed from his backpack, drenching Crabbe, Finnegan, and C.J., before drying the nearby woods. Finnegan and C.J., in the meantime, were lazing at the bottom of the bed. Finnegan\'s muffler was pulled up over his head. He couldn\'t hear anything, but his hand was moving as if he was singing. "Buddy, buddy, look, just follow my voice!" his shoulder was up again and he was picked up.\n\nAs they were carried to the car, Moisture\'s head was smashed through the campfire wall with a dull thud. "Checkmate!"\n\nOutside, the bitter wind blew, and in the distance, another dawn owl hooted above the trees. Crabbe wasn\'t surprised he didn\'t understand, since he was a mere teen. Finnegan was always remarkable for that.\n\n"What was that?"\n\nRaze had finished assembling his gift in the back of the car. "I, uh… found that overly-enthusiastic girl in the proximity of burning cadavers. Probably should rethink this again." He handed it to Crabbe, hefting it tightly on his backseat as he went to get their luggage. "Here. Jaune Arc, don\'t be too much trouble."\n\n"Sorry," he whispered. He placed the snip of Fur on the ring of honor, and then of course, he put on his shirt. Flipping on the lights, they rolled into Beacon to check into their room.\n\n"Hey Raze," C.J. greeted, placing her backpack on his desk. "How\'d you like to bring me over for breakfast?"\n\nAnswering, "Fine," he put the brown bag and eggs down from his lap on the desk and turned his attentions to her. A hand gently pulled her around. "I was thinking we might try something with milk."\n\n"Wow," C.J. crackled as she groaned. "Eggs aren\'t that bad I guess…"\n\n"I\'ve noticed," he admitted. "But it\'s impossible to get them wet at this hour. We\'ll just have to wait for dinnertime."']
generation of 512 tokens took 10.469078063964844 seconds
"""