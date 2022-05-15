import torch
from torch.distributions import Categorical

from agents import base


def categorical_distribution(logits: torch.Tensor) -> torch.distributions.Categorical:
    """Returns categorical distribution that support sample(), entropy(), and log_prob()."""
    return Categorical(logits=logits)


def categorical_importance_sampling_ratios(
    pi_logits_t: torch.Tensor, mu_logits_t: torch.Tensor, a_t: torch.Tensor
) -> torch.Tensor:
    """Compute importance sampling ratios from logits.

    Args:
      pi_logits_t: unnormalized logits at time t for the target policy,
        shape [B, num_actions] or [T, B, num_actions].
      mu_logits_t: unnormalized logits at time t for the behavior policy,
        shape [B, num_actions] or [T, B, num_actions].
      a_t: actions at time t, shape [B] or [T, B].

    Returns:
      importance sampling ratios, shape [B] or [T, B].
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(pi_logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(mu_logits_t, (2, 3), torch.float32)
    base.assert_rank_and_dtype(a_t, (1, 2), torch.long)

    pi_m = Categorical(logits=pi_logits_t)
    mu_m = Categorical(logits=mu_logits_t)

    pi_logprob_a_t = pi_m.log_prob(a_t)
    mu_logprob_a_t = mu_m.log_prob(a_t)

    return torch.exp(pi_logprob_a_t - mu_logprob_a_t)
