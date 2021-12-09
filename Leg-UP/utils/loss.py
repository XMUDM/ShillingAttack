import torch
import torch.nn.functional as F

from utils.utils import EPSILON

__all__ = ["mse_loss", "mult_ce_loss", "binary_ce_loss", "kld_loss",
           "sampled_bce_loss", "sampled_cml_loss"]

"""Model training losses."""
bce_loss = torch.nn.BCELoss(reduction='none')


def mse_loss(data, logits, weight):
    """Mean square error loss."""
    weights = torch.ones_like(data)
    weights[data > 0] = weight
    res = weights * (data - logits) ** 2
    return res.sum(1)


def mult_ce_loss(data, logits):
    """Multi-class cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs * data

    instance_data = data.sum(1)
    instance_loss = loss.sum(1)
    # Avoid divide by zeros.
    res = instance_loss / (instance_data + EPSILON)
    return res


def binary_ce_loss(data, logits):
    """Binary-class cross-entropy loss."""
    return bce_loss(torch.sigmoid(logits), data).mean(1)


def kld_loss(mu, log_var):
    """KL-divergence."""
    return -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def sampled_bce_loss(logits, n_negatives):
    """Binary-class cross-entropy loss with sampled negatives."""
    pos_logits, neg_logits = torch.split(logits, [1, n_negatives], 1)
    data = torch.cat([
        torch.ones_like(pos_logits), torch.zeros_like(neg_logits)
    ], 1)
    return bce_loss(torch.sigmoid(logits), data).mean(1)


def sampled_cml_loss(distances, n_negatives, margin):
    """Hinge loss with sampled negatives."""
    # Distances here are the negative euclidean distances.
    pos_distances, neg_distances = torch.split(-distances, [1, n_negatives], 1)
    neg_distances = neg_distances.min(1).values.unsqueeze(-1)
    res = pos_distances - neg_distances + margin
    res[res < 0] = 0
    return res.sum(1)
