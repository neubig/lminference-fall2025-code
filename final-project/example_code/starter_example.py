# modal_hf_openai_api.py
import modal

app = modal.App("andrewID-1")

# Define the image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "accelerate",
    "fastapi[standard]", 
)


@app.cls(
    image=image,
    gpu="A100-80GB:2",  
    scaledown_window=600, # allow 10 min after the last request
)

@modal.concurrent(max_inputs=300)
class Model:
    @modal.enter()
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "Qwen/Qwen3-0.6B"  # Use Qwen models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )
    
    @modal.fastapi_endpoint(method="POST")  
    def completions(self, request: dict):
        import torch
        
        # Extract OpenAI-style parameters
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)
        
        # Handle both single prompt (string) and batch (list)
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        # Batch tokenization
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Generate for all prompts in batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
        
        # Decode all outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Create choices for each completion
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, (generated_text, input_ids) in enumerate(zip(generated_texts, inputs.input_ids)):
            prompt_tokens = len(input_ids)
            completion_tokens = len(outputs[i]) - prompt_tokens
            
            if outputs[i][-1].item() == self.tokenizer.eos_token_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"
            
            choices.append({
                "text": generated_text,
                "index": i,
                "finish_reason": finish_reason
            })
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
        
        # Return OpenAI-style response
        return {
            "choices": choices,
            "model": "andrewID-system-1",
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }


