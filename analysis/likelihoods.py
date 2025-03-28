# analysis/likelihoods.py

import torch
import torch.nn.functional as F
from typing import Dict


def compute_likelihoods(output: Dict) -> Dict:
    """
    Compute log-likelihoods and entropy from logits and input_ids.
    Assumes logits and input_ids are returned from run_generation().
    """
    logits = torch.stack(output["scores"]).squeeze(1)  # [seq_len, vocab_size]
    input_ids = output["generated_ids"]
    vocab_size = logits.shape[-1]

    # Align shapes: remove the last token from logits, and first token from targets
    shifted_logits = logits[:-1]  # [seq_len - 1, vocab_size]
    shifted_labels = input_ids[1:]  # [seq_len - 1]

    log_probs = F.log_softmax(shifted_logits, dim=-1)  # [seq_len - 1, vocab_size]

    if torch.any(shifted_labels >= log_probs.shape[1]) or torch.any(shifted_labels < 0):
        print("Error: Shifted labels are out of bounds!")
        print(f"shifted_labels: {shifted_labels}")
        print(f"log_probs shape: {log_probs.shape}")
        raise ValueError("Invalid indices in shifted_labels.")

    # Token-wise log-likelihood (negative NLL loss)
    token_log_likelihoods = log_probs[range(len(shifted_labels)), shifted_labels]

    # Token-wise entropy
    entropy_per_token = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)

    return {
        "token_log_likelihoods": token_log_likelihoods.detach().cpu(),
        "entropy_per_token": entropy_per_token.detach().cpu(),
        "logits": logits.detach().cpu(),
        "input_ids": input_ids.detach().cpu()
    }
