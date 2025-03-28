# analysis/uncertainty.py

import torch
from typing import Dict
from sentence_transformers import SentenceTransformer, util

# Cache SBERT model to avoid reloading
_sbert_cache = {}

def compute_uncertainty_scores(likelihood_dict: Dict, output: Dict, methods: list = ['entropy', 'lastde', 'lastn_entropy', 'logit_gap', 'attentionsar', 'bertsar']) -> Dict:
    """
    Compute scalar uncertainty scores from token-level likelihoods for multiple methods.

    Args:
        likelihood_dict: Dict with keys like 'token_log_likelihoods', 'entropy_per_token', etc.
        output: Dict with model output (e.g., generated text, attention weights)
        methods: List of methods to compute, e.g., ['entropy', 'lastde', 'logit_gap', 'attentionsar', 'bertsar']

    Returns:
        Dict with keys corresponding to the methods, each with its computed score.
    """
    scores = {}

    for method in methods:
        try:
            if method == 'entropy':
                scores['entropy'] = likelihood_dict['entropy_per_token'].mean().item()

            elif method == 'lastde':
                ent = likelihood_dict['entropy_per_token']
                scores['lastde'] = ent[-1].item() if ent.numel() > 0 else float("nan")

            elif method == 'lastn_entropy':
                n = 10  # Number of tokens for lastn_entropy, this can be passed as a parameter if needed
                ent = likelihood_dict['entropy_per_token']
                scores['lastn_entropy'] = ent[-n:].mean().item() if ent.numel() >= n else ent.mean().item()

            elif method == 'logit_gap':
                logits = likelihood_dict['logits']
                if isinstance(logits, tuple):
                    logits = logits[0]
                if logits.dim() == 3:
                    logits = logits[0]
                if logits.dim() != 2:
                    print(f"⚠️ Unexpected logits shape: {logits.shape}")
                    scores['logit_gap'] = float("nan")
                    continue
                topk = torch.topk(logits, k=2, dim=-1).values
                gaps = topk[:, 0] - topk[:, 1]
                scores['logit_gap'] = -gaps.mean().item()
            elif method == 'attentionsar':
                scores['attentionsar'] = compute_attentionsar_uncertainty(likelihood_dict, output)

            elif method == 'bertsar':
                scores['bertsar'] = compute_bert_sar_uncertainty(likelihood_dict, output)

            else:
                print(f"⚠️ Unknown uncertainty method: {method}")
                scores[method] = float("nan")

        except IndexError as e:
            print(f"⚠️ Skipping IndexError: {e}")
            scores[method] = float("nan")  # Return NaN if there's an index error

    return scores

# Helper functions for attentionsar and bertsar
def compute_attentionsar_uncertainty(likelihoods, output):
    log_likelihoods = likelihoods["token_log_likelihoods"]
    if log_likelihoods.numel() == 0:
        return float("nan")

    attentions = output.get("log_attentions", None)
    if not attentions or not isinstance(attentions, list):
        print("⚠️ No valid attention scores for SAR. Falling back to mean log-likelihood.")
        return -log_likelihoods.mean().item()

    # Use attention from the last layer
    last_layer_attn = attentions[-1]  # shape: [batch, heads, tgt_len, src_len]
    if isinstance(last_layer_attn, tuple):
        last_layer_attn = last_layer_attn[0]

    if last_layer_attn.ndim != 4:
        print("⚠️ Unexpected attention shape:", last_layer_attn.shape)
        return -log_likelihoods.mean().item()

    # Average over heads and get attention on generated tokens only
    attn_weights = last_layer_attn.mean(dim=1).mean(dim=0)  # [tgt_len, src_len]

    # Take the diagonal as a proxy for self-relevance (token attends to itself)
    diag_attn = torch.diagonal(attn_weights, dim1=0, dim2=1)  # shape: [seq_len]

    # Normalize attention weights
    attn_weights = diag_attn[:len(log_likelihoods)]
    attn_weights = attn_weights / attn_weights.sum()

    # Compute SAR as attention-weighted negative log-likelihood
    sar_score = -(log_likelihoods[:len(attn_weights)] * attn_weights).sum().item()

    return sar_score

def compute_bert_sar_uncertainty(likelihoods, output, model_name="all-MiniLM-L6-v2"):
    log_likelihoods = likelihoods["token_log_likelihoods"]
    if log_likelihoods.numel() == 0:
        return float("nan")

    if model_name not in _sbert_cache:
        _sbert_cache[model_name] = SentenceTransformer(model_name)
    sbert = _sbert_cache[model_name]

    try:
        # Get the tokens (optional fallback)
        input_text = output.get("input_text", "")
        answer_text = output.get("generated_text", "")

        # Tokenize into individual words (rough approximation)
        tokens = input_text.split()
        token_embeddings = sbert.encode(tokens, convert_to_tensor=True, normalize_embeddings=True)
        answer_embedding = sbert.encode(answer_text, convert_to_tensor=True, normalize_embeddings=True)

        # Compute cosine similarity between each token and the answer
        similarities = util.pytorch_cos_sim(token_embeddings, answer_embedding).squeeze()

        # Normalize
        similarities = similarities[:len(log_likelihoods)]
        weights = similarities / similarities.sum().clamp(min=1e-6)

        # Weighted negative log-likelihood
        score = -(log_likelihoods[:len(weights)] * weights).sum().item()

        return score
    except Exception as e:
        print(f"⚠️ BERT-SAR error: {e}")
        return float("nan")
