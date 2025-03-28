# analysis/uncertainty.py

import torch
from typing import Dict


def compute_uncertainty_scores(likelihood_dict: Dict, method: str = 'lastde') -> Dict:
    """
    Compute scalar uncertainty score from token-level likelihoods.

    Args:
        likelihood_dict: Dict with keys like 'token_log_likelihoods', 'entropy_per_token', etc.
        method: which metric to use: 'lastde', 'entropy', 'logit_gap', etc.

    Returns:
        Dict with a single key: 'score', and optionally intermediates.
    """
    if method == 'entropy':
        score = likelihood_dict['entropy_per_token'].mean()
    elif method == 'lastde':
        # Last token entropy (tail token)
        score = likelihood_dict['entropy_per_token'][-1]
    elif method == 'logit_gap':
        logits = likelihood_dict['logits']
        if logits.dim() == 3:
            logits = logits[0]  # [seq_len, vocab] if wrapped

        # logit gap = top1 - top2 per token
        topk = torch.topk(logits, k=2, dim=-1).values  # [seq_len, 2]
        gaps = topk[:, 0] - topk[:, 1]
        score = -gaps.mean()  # larger gap = more confident, so we invert
    else:
        raise NotImplementedError(f"Unknown uncertainty method: {method}")

    return {"score": score.item()}