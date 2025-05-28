import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from analysis.uncertainty import compute_uncertainty_scores, compute_aware_uncertainty

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
    embedding_matrix = model.get_input_embeddings().weight.detach().to(torch.float32)

    result = []
    for i, sample in enumerate(batch):
        item = {
            "input_ids": encoded["input_ids"][i],
            "generated_ids": generated[i],
            "generated_text": decoded[i],
        }

        if return_logits:
            item["scores"] = outputs.scores

        if return_attentions and hasattr(outputs, 'attentions'):
            log_attention = []
            for attention in outputs.attentions:
                attention_tensor = attention[0] if isinstance(attention, tuple) else attention
                log_attention.append(torch.log(1 + torch.clamp(attention_tensor, min=1e-10)))
            item["log_attentions"] = log_attention

        # === Full forward pass ===
        full_input_ids = torch.cat([
            encoded["input_ids"][i],
            generated[i][encoded["input_ids"].shape[1]:]
        ]).unsqueeze(0).to(model.device)

        with torch.no_grad():
            forward_out = model(input_ids=full_input_ids, output_attentions=True, return_dict=True)

        logits = forward_out.logits.squeeze(0).to(torch.float32)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_ids = full_input_ids[0][1:]  # shift to next-token prediction
        token_log_likelihoods = log_probs[:-1, :].gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
        entropy_per_token = (-log_probs * log_probs).sum(dim=-1)
        attentions = [a.to(torch.float32) for a in forward_out.attentions]

        # === Get question embedding ===
        question_tokens = tokenizer(sample["question"], return_tensors="pt", add_special_tokens=False).to(model.device)
        question_embedding = embedding_matrix[question_tokens["input_ids"].squeeze(0)].mean(dim=0)

        # === Unified uncertainty scoring ===
        likelihood_dict = {
            "token_log_likelihoods": token_log_likelihoods,
            "entropy_per_token": entropy_per_token,
            "logits": logits
        }

        model_output = {
            "generated_text": decoded[i],
            "input_text": prompts[i],
            "log_attentions": attentions,
            "embedding_matrix": embedding_matrix,
            "question_embedding": question_embedding
        }

        try:
            scores = compute_uncertainty_scores(
                likelihood_dict, model_output, methods=uncertainty_methods or []
            )
            item["uncertainty_scores"] = scores
        except Exception as e:
            print(f"⚠️ Failed to compute uncertainty scores for sample {i}: {e}")
            item["uncertainty_scores"] = {}

        result.append(item)

    return result

