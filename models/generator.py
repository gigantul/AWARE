import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

_model_cache = {}
_tokenizer_cache = {}

def load_model_and_tokenizer(model_name: str):
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer
    return _model_cache[model_name], _tokenizer_cache[model_name]

def run_generation(batch: List[Dict], model_name: str, return_logits: bool = True, return_attentions: bool = False) -> List[Dict]:
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Prepare the prompts
    prompts = [sample.get("prompt", f"Question: {sample['question']} Answer:") for sample in batch]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        # Generate outputs from the model
        outputs = model.generate(
            **encoded,
            max_length=encoded["input_ids"].shape[1] + 64,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=return_logits,
            output_attentions=return_attentions
        )

    # Extract generated sequences and decode them
    generated = outputs.sequences
    decoded = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)

    result = []
    for i, sample in enumerate(batch):
        item = {
            "input_ids": encoded["input_ids"][i],
            "generated_ids": generated[i],
            "generated_text": decoded[i]
        }
        if return_logits:
            item["scores"] = outputs.scores  # token-wise logits per position

        if return_attentions and hasattr(outputs, 'attentions'):
            attentions = outputs.attentions
            # Apply logarithmic transformation to attention weights safely
            log_attention = []
            for attention in attentions:
                if isinstance(attention, torch.Tensor):
                    # Clamp to avoid log(0) or negative values
                    attention = torch.clamp(attention, min=1e-10)  # Avoid negative or zero values
                    log_attention.append(torch.log(1 + attention))  # Apply log-normal scale
                else:
                    # If attention is a tuple, unpack and then apply transformation
                    attention_tensor = torch.clamp(attention[0], min=1e-10)
                    log_attention.append(torch.log(1 + attention_tensor))  # Assuming attention[0] is the tensor
            item["log_attentions"] = log_attention

        result.append(item)

    return result

