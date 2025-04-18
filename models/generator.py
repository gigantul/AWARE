import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from analysis.uncertainty import compute_uncertainty_scores, compute_aware_uncertainty  # Adjust if your path is different

_model_cache = {}
_tokenizer_cache = {}


def load_model_and_tokenizer(model_name: str):
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(

            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Force eager to suppress SDPA warnings for OPT
        )
        model.eval()
        _model_cache[model_name] = model
        _tokenizer_cache[model_name] = tokenizer
    return _model_cache[model_name], _tokenizer_cache[model_name]

def run_generation(
    batch: List[Dict],
    model_name: str,
    return_logits: bool = True,
    return_attentions: bool = False,
    uncertainty_methods: List[str] = None
) -> List[Dict]:

    model, tokenizer = load_model_and_tokenizer(model_name)

    # Prepare the prompts
    prompts = [sample.get("prompt", f"Question: {sample['question']} Answer:") for sample in batch]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=encoded["input_ids"].shape[1] + 64,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=return_logits,
            output_attentions=return_attentions
        )

    generated = outputs.sequences
    decoded = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)

    for i, (q, a) in enumerate(zip(questions, decoded)):
        print(f"\n[Sample {i}]")
        print(f"Q: {q}")
        print(f"A: {a}")
        print("-" * 50)

    result = []
    for i, sample in enumerate(batch):
        item = {
            "input_ids": encoded["input_ids"][i],
            "generated_ids": generated[i],
            "generated_text": decoded[i],
        }

        if return_logits:
            item["scores"] = outputs.scores  # list of [batch, vocab] scores per generated token

        if return_attentions and hasattr(outputs, 'attentions'):
            attentions = outputs.attentions
            log_attention = []
            for attention in attentions:
                if isinstance(attention, torch.Tensor):
                    attention = torch.clamp(attention, min=1e-10)
                    log_attention.append(torch.log(1 + attention))
                else:
                    attention_tensor = torch.clamp(attention[0], min=1e-10)
                    log_attention.append(torch.log(1 + attention_tensor))
            item["log_attentions"] = log_attention

        # Add context for baseline uncertainty score computation
        likelihood_dict = {
            "token_log_likelihoods": sample.get("token_log_likelihoods", torch.tensor([])),
            "entropy_per_token": sample.get("entropy_per_token", torch.tensor([])),
            "logits": outputs.scores if return_logits else None
        }

        model_output = {
            "generated_text": decoded[i],
            "input_text": prompts[i],
            "log_attentions": item.get("log_attentions", None),
            "embedding_matrix": model.get_input_embeddings().weight.detach()
        }

        scores = {}
        if uncertainty_methods:
            try:
                # Baseline scores
                baseline_methods = [m for m in uncertainty_methods if m != "aware"]
                scores.update(compute_uncertainty_scores(likelihood_dict, model_output, methods=baseline_methods))

                # AWARE score via full forward
                if "aware" in uncertainty_methods:
                    full_input_ids = generated[i].unsqueeze(0).to(model.device)
                    with torch.no_grad():
                        forward_out = model(
                            input_ids=full_input_ids,
                            output_attentions=True,
                            return_dict=True
                        )
                        full_logits = forward_out.logits.squeeze(0).to(torch.float32)
                        full_attentions = [a.to(torch.float32) for a in forward_out.attentions]
                        embedding_matrix = model.get_input_embeddings().weight.detach().to(full_logits.device).to(torch.float32)

                    aware_likelihood_dict = {
                        "logits": full_logits,
                        "token_log_likelihoods": None,
                        "entropy_per_token": None
                    }
                    aware_output = {
                        "generated_text": decoded[i],
                        "input_text": prompts[i],
                        "log_attentions": full_attentions,
                        "embedding_matrix": embedding_matrix
                    }
                    scores["aware"] = compute_aware_uncertainty(aware_likelihood_dict, aware_output)

                item["uncertainty_scores"] = scores
            except Exception as e:
                print(f"⚠️ Failed to compute uncertainty scores for sample {i}: {e}")
                item["uncertainty_scores"] = {}

        result.append(item)

    return result
